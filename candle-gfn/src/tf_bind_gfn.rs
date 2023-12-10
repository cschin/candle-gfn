
use crate::model::ModelTrait;
use crate::sampler::{MDPTrait, Sampler, Sampling, SamplingConfiguration, MDP};
use crate::state::{State, StateCollection, StateIdType, StateTrait};
use crate::trajectory::Trajectory;
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use fxhash::FxHashMap;
use rand::{self, Rng};

pub type MerStateDateType = Vec<u8>;
pub type MerState = State<MerStateDateType>;

#[derive(Clone)]
pub struct TFParameters<'a> {
    pub size: u32,
    pub number_trajectories: u32,
    pub rewards: FxHashMap<Vec<u8>, f32>,
    pub device: &'a Device,
}

pub struct TFModel<'a, P> {
    ln1: Linear,
    ln2: Linear,
    f0: Tensor,
    pub varmap: VarMap,
    parameter: &'a P,
}

impl<'a> TFModel<'a, TFParameters<'a>> {
    pub fn new(parameter: &'a TFParameters) -> Result<Self> {
        let device = parameter.device;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let in_d: usize = (parameter.size * 3) as usize;
        let out_d = 2_usize; // a simple score of log p
        let ln1 = candle_nn::linear(in_d, 128, vb.pp("ln1"))?;
        let ln2 = candle_nn::linear(128, out_d, vb.pp("ln2"))?;
        let f0 = vb
            .get_with_hints((1, 1), "f0", candle_nn::Init::Const(5.0))
            .unwrap();
        Ok(Self {
            ln1,
            ln2,
            f0,
            varmap,
            parameter,
        })
    }
}

impl<'a> ModelTrait<MerStateDateType> for TFModel<'a, TFParameters<'a>> {
    fn forward_ss_flow(
        &self,
        source: &impl StateTrait<MerStateDateType>,
        sink: &impl StateTrait<MerStateDateType>,
    ) -> Result<Tensor> {
        let source_tensor = source.get_tensor().unwrap();
        let sink_tensor = sink.get_tensor().unwrap();
        let device = self.parameter.device;

        let source_out = self.ln1.forward(source_tensor)?.detach()?;
        let source_out = source_out.relu()?;
        let source_out = self.ln2.forward(&source_out)?.exp()?;

        let sink_out = self.ln1.forward(sink_tensor)?.detach()?;
        let sink_out = sink_out.relu()?;
        let sink_out = self.ln2.forward(&sink_out)?.exp()?;

        let idx_0 = Tensor::new(&[0u32; 1], device)?;
        let idx_1 = Tensor::new(&[1u32; 1], device)?;
        let source_flow = source_out.index_select(&idx_0, 1)?;
        let source_score = source_out.index_select(&idx_1, 1)?;
        let sink_score = sink_out.index_select(&idx_1, 1)?;

        let out_tensor = source_score.sub(&sink_score)?.add(&source_flow)?.exp()?;
        Ok(out_tensor)
    }

    fn get_f0(&self) -> Result<&Tensor> {
        Ok(&self.f0)
    }

    fn forward_ss_flow_batch_with_states(
        &self,
        ss_pairs: &[(
            &impl StateTrait<MerStateDateType>,
            &impl StateTrait<MerStateDateType>,
        )],
        batch_size: usize,
    ) -> Result<Tensor> {
        let device = self.parameter.device;

        let mut out_tensors = Vec::<_>::new();
        //let mut out_tensors = Vec::<_>::new();
        (0..ss_pairs.len())
            .step_by(batch_size)
            .try_for_each(|start_idx| -> Result<()> {
                let end_idx = if start_idx + batch_size < ss_pairs.len() {
                    start_idx + batch_size
                } else {
                    ss_pairs.len()
                };
                let mut source_tensors = Vec::<&Tensor>::new();
                let mut sink_tensors = Vec::<&Tensor>::new();

                ss_pairs[start_idx..end_idx].iter().for_each(|&(s, t)| {
                    source_tensors.push(s.get_tensor().unwrap());
                    sink_tensors.push(t.get_tensor().unwrap());
                });

                let source_tensors = Tensor::cat(&source_tensors[..], 0).unwrap();
                let sink_tensors = Tensor::cat(&sink_tensors[..], 0).unwrap();

                let source_out = self.ln1.forward(&source_tensors)?.detach()?;
                let source_out = source_out.relu()?;
                let source_out = self.ln2.forward(&source_out)?.exp()?;

                let sink_out = self.ln1.forward(&sink_tensors)?.detach()?;
                let sink_out = sink_out.relu()?;
                let sink_out = self.ln2.forward(&sink_out)?.exp()?;

                let idx_0 = Tensor::new(&[0u32; 1], device)?;
                let idx_1 = Tensor::new(&[1u32; 1], device)?;
                let source_flow = source_out.index_select(&idx_0, 1)?;
                let source_score = source_out.index_select(&idx_1, 1)?;
                let sink_score = sink_out.index_select(&idx_1, 1)?;

                let out_tensor = source_score.sub(&sink_score)?.add(&source_flow)?.exp()?;
                out_tensors.push(out_tensor);

                Ok(())
            })?;
        Ok(Tensor::cat(&out_tensors, 0)?)
    }

    fn reverse_ss_flow_batch(
        &self,
        _ss_p: &[(StateIdType, StateIdType)],
        _batch_size: usize,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn reverse_ss_flow(
        &self,
        _source: &impl StateTrait<MerStateDateType>,
        _sink: &impl StateTrait<MerStateDateType>,
    ) -> Result<Tensor> {
        unimplemented!()
    }
}

impl MerState {
    pub fn new(
        id: StateIdType,
        mer: &[u8],
        is_terminal: bool,
        reward: f32,
        parameter: &TFParameters,
    ) -> Self {
        let size = parameter.size as usize;
        let mut t = vec![0.0f32; size * 3];
        mer.iter().enumerate().for_each(|(i, &b)| {
            t[i * 3] = 1.0;
            match b {
                0 => {
                    t[i * 3 + 1] = 0.0;
                    t[i * 3 + 2] = 0.0;
                }
                1 => {
                    t[i * 3 + 1] = 1.0;
                    t[i * 3 + 2] = 0.0;
                }
                2 => {
                    t[i * 3 + 1] = 0.0;
                    t[i * 3 + 2] = 1.0;
                }
                3 => {
                    t[i * 3 + 1] = 1.0;
                    t[i * 3 + 2] = 1.0;
                }
                _ => (),
            }
        });
        let tensor =
            Tensor::from_vec(t, &[1, size * 3], parameter.device).expect("create tensor fail");
        Self {
            id,
            data: Vec::from(mer),
            tensor,
            is_terminal,
            reward,
        }
    }

    pub fn set_reward(&mut self, reward: f32) {
        self.reward = reward;
    }
}

pub fn get_id_from_mer(mer: &[u8]) -> u32 {
    let mut id = 0u32;
    mer.iter().for_each(|b| {
        id <<= 2;
        id |= (b & 0b11) as u32;
    });
    id <<= 4;
    id |= (mer.len() & 0b1111) as u32;
    id
}

impl StateTrait<MerStateDateType> for MerState {
    fn get_id(&self) -> u32 {
        self.id
    }

    fn get_tensor(&self) -> Result<&Tensor> {
        Ok(&self.tensor)
    }

    fn get_forward_flow(
        &self,
        next_state: &impl StateTrait<MerStateDateType>,
        model: &impl ModelTrait<MerStateDateType>,
    ) -> Result<Tensor> {
        model.forward_ss_flow(self, next_state)
    }

    fn get_previous_flow(
        &self,
        previous_state: &impl StateTrait<MerStateDateType>,
        model: &impl ModelTrait<MerStateDateType>,
    ) -> Result<Tensor> {
        model.forward_ss_flow(previous_state, self)
    }

    fn get_data(&self) -> MerStateDateType {
        self.data.clone()
    }
}

pub type MerStateCollection = StateCollection<StateIdType, MerState>;

pub type MerMDP = MDP<StateIdType, MerState>;

impl<'a> MDPTrait<StateIdType, MerState, TFParameters<'a>> for MerMDP {
    fn mdp_next_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut MerStateCollection,
        parameters: &TFParameters,
    ) -> Option<Vec<StateIdType>> {
        let state_data = collection
            .map
            .get(&state_id)
            .expect("can get the stat")
            .get_data();

        if state_data.len() == parameters.size as usize {
            return None;
        }

        let mut next_states = Vec::<StateIdType>::new();

        (0..4).for_each(|base| {
            let mut new_state = Vec::<u8>::new();
            new_state.push(base);
            new_state.extend(&state_data);
            let new_state_id = get_id_from_mer(&new_state);
            if collection.map.contains_key(&new_state_id) {
                next_states.push(new_state_id);
            } else {
                let new_state = Box::new(MerState::new(
                    new_state_id,
                    &new_state,
                    false,
                    0.0,
                    parameters,
                ));
                collection.map.entry(new_state_id).or_insert(new_state);
                next_states.push(new_state_id);
            }

            let mut new_state = Vec::<u8>::new();
            new_state.extend(&state_data);
            new_state.push(base);
            let new_state_id = get_id_from_mer(&new_state);
            if collection.map.contains_key(&new_state_id) {
                next_states.push(new_state_id);
            } else {
                let new_state = Box::new(MerState::new(
                    new_state_id,
                    &new_state,
                    false, // temp hack 
                    0.0,
                    parameters,
                ));
                collection.map.entry(new_state_id).or_insert(new_state);
                next_states.push(new_state_id);
            }
        });


        if next_states.is_empty() {
            None
        } else {
            Some(next_states)
        }
    }

    fn mdp_next_one_uniform(
        &self,
        state_id: StateIdType,
        collection: &mut MerStateCollection,
        parameters: &TFParameters,
    ) -> Option<StateIdType> {
        let mut rng = rand::thread_rng();
        self.mdp_next_possible_states(state_id, collection, parameters)
            .map(|states| states[rng.gen::<usize>().rem_euclid(states.len())])
    }

    fn mdp_previous_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut MerStateCollection,
        _parameters: &TFParameters,
    ) -> Option<Vec<StateIdType>> {
        let _state = collection.map.get(&state_id).expect("can get the stat");
        unimplemented!()
    }

    fn mdp_previous_one_uniform(
        &self,
        state_id: StateIdType,
        collection: &mut MerStateCollection,
        parameters: &TFParameters,
    ) -> Option<StateIdType> {
        let mut rng = rand::thread_rng();
        self.mdp_previous_possible_states(state_id, collection, parameters)
            .map(|states| states[rng.gen::<usize>().rem_euclid(states.len())])
    }
}

pub type TFSampler = Sampler<StateIdType>;
pub type TFSamplingConfiguration<'a> = SamplingConfiguration<
    'a,
    StateIdType,
    MerState,
    TFModel<'a, TFParameters<'a>>,
    TFParameters<'a>,
>;

impl<'a> Sampling<StateIdType, TFSamplingConfiguration<'a>> for TFSampler {
    fn sample_a_new_trajectory(
        &mut self,
        config: &mut TFSamplingConfiguration,
    ) -> Trajectory<StateIdType> {
        let begin = config.begin;
        let collection = &mut config.collection;
        let model = config.model;
        let mdp = &mut config.mdp;
        let parameters = config.parameters;

        let mut traj = Trajectory::new();

        loop {
            traj.clear();
            let mut state_id = begin;
            traj.push(state_id);
            let model = model.unwrap();
            let mut rng = rand::thread_rng();
            while let Some(next_states) =
                mdp.mdp_next_possible_states(state_id, collection, parameters)
            {
                let state_and_flow = next_states
                    .into_iter()
                    .map(|sid| {
                        let s0 = collection.map.get(&state_id).unwrap().as_ref();
                        let s1 = collection.map.get(&sid).unwrap().as_ref();
                        let p = model
                            .forward_ss_flow(s0, s1)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .get(0)
                            .unwrap();
                        //println!("{:?}", p);
                        (sid, p.to_scalar::<f32>().unwrap())
                    })
                    .collect::<Vec<_>>();
                if state_and_flow.is_empty() {
                    break;
                };

                let mut sump: f32 = state_and_flow.iter().map(|v| v.1).sum();
                let state = collection.map.get(&state_id).unwrap().as_ref();
                if state.is_terminal {
                    sump += state.reward;
                }

                let mut acc_sump = Vec::new();
                //let epsilon: f32 = 0.05;
                let mut th = 0.0f32;
                state_and_flow.iter().for_each(|(id, p)| {
                    th += p / sump;
                    acc_sump.push((*id, th))
                });
                let mut t = rng.gen::<f32>();
                let mut updated = false;
                if state.is_terminal {
                    if t < state.reward / sump {
                        break;
                    } else {
                        t -= state.reward / sump;
                    }
                }
                //println!("acc_sump: {:?} {}", acc_sump, t);
                for (id, th) in acc_sump {
                    if t < th {
                        state_id = id;
                        traj.push(state_id);
                        updated = true;
                        break;
                    }
                }
                if !updated {
                    let d = state_and_flow[rng.gen::<usize>().rem_euclid(state_and_flow.len())];
                    state_id = d.0;
                    traj.push(state_id);
                }
            }
            let state = collection.map.get(&state_id).unwrap().as_ref();
            if state.is_terminal {
                break;
            };
        }
        //println!("traj: {:?}", traj.trajectory);
        traj
    }

    fn sample_trajectories(&mut self, config: &mut TFSamplingConfiguration) {
        let number_trajectories = config.parameters.number_trajectories;
        (0..number_trajectories).for_each(|_| {
            let trajectory = self.sample_a_new_trajectory(config);
            // trajectory
            //     .get_parent_offspring_pairs()
            //     .iter()
            //     .for_each(|&(s0id, s1id)| {
            //         self.offsprings.entry(s0id).or_default().push(s1id);
            //         self.parents.entry(s1id).or_default().push(s0id);
            //     });
            self.trajectories.push(trajectory);
        });
    }
}

#[cfg(test)]
mod tests {


    use crate::sampler::MDPTrait;

    use super::*;
    #[test]
    fn test_tf_mdp() -> Result<()> {
        let parameters = TFParameters {
            size: 8,
            number_trajectories: 20,
            rewards: FxHashMap::default(),
            device: &candle_core::Device::Cpu,
        };
        let mer = vec![0, 1, 2u8];
        let mer_id = get_id_from_mer(&mer);
        let m_state = MerState::new(mer_id, &mer, false, 0.0, &parameters);
        let model = TFModel::new(&parameters).unwrap();

        let collection = &mut MerStateCollection::default();
        collection.map.insert(mer_id, Box::new(m_state));
        let mdp = MerMDP::new();
        let mut state_id = mer_id;
        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection, &parameters) {
            println!("state_id: {:?} {}", state_id, state_id & 0b1111);
            let state_from = collection.map.get(&state_id).unwrap().as_ref();
            let state_to = collection.map.get(&s).unwrap().as_ref();

            let out = model
                .forward_ss_flow(state_from, state_to)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            println!("1: {:?}", out);

            let ss_pairs = vec![(state_from, state_to), (state_from, state_to)];

            let out = model
                .forward_ss_flow_batch_with_states(&ss_pairs, 2)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            println!("2: {:?}", out);

            state_id = s;
        }
        println!("state_id: {:?} {}", state_id, state_id & 0b1111);
        Ok(())
    }

    #[test]
    fn generate_trajectory() {

        use crate::sampler::Sampling;

        use super::*;
        use candle_core::Device;
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).expect("no cuda device available")
        } else {
            Device::Cpu
        };
        let device = &device;
        let parameters = &TFParameters {
            size: 8,
            number_trajectories: 20,
            rewards: FxHashMap::default(),
            device: &candle_core::Device::Cpu,
        };
        let mer: Vec<u8> = vec![];
        let mer_id = get_id_from_mer(&mer);
        let m_state = MerState::new(mer_id, &mer, false, 0.0, parameters);
        let model = TFModel::new(parameters).unwrap();

        let collection = &mut MerStateCollection::default();
        collection.map.insert(mer_id, Box::new(m_state));
        let mdp = &mut MerMDP::new();

        (0..(1<<16) as u32).for_each(|n| {
            let state = (0..8).map(|i|{
                ((n >> (i*2)) & 0b11) as u8
            }).collect::<Vec<u8>>();
            let new_state_id = get_id_from_mer(&state);
            let new_state = Box::new(MerState::new(
                new_state_id,
                &state,
                true,
                0.0,
                parameters,
            ));
            collection.map.entry(new_state_id).or_insert(new_state);
        }); 

        let mut config = TFSamplingConfiguration {
            begin: 0,
            collection,
            mdp,
            model: Some(&model),
            device,
            parameters,
        };

        let mut sampler = TFSampler::new();
        sampler.trajectories.clear();
        sampler.sample_trajectories(&mut config);
        println!("{:?}", sampler.trajectories.len());

    }

}

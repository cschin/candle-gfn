use crate::model::ModelTrait;
use crate::sampler::{MDPTrait, Sampler, Sampling, SamplingConfiguration, MDP};
use crate::state::{State, StateCollection, StateIdType, StateTrait};
use crate::trajectory::Trajectory;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{self, Rng};

#[derive(Clone)]
pub struct SimpleGridParameters<'a> {
    pub max_x: u32,
    pub max_y: u32,
    pub number_trajectories: u32,
    pub rewards: Vec<((u32, u32), f32)>,
    pub device: &'a Device,
}

pub struct SimpleGridModel<'a, P> {
    //ln1: Linear,
    //ln2: Linear,
    f0: Tensor,
    values: Tensor,
    f: Tensor,
    pub varmap: VarMap,
    parameter: &'a P,
}

impl<'a> SimpleGridModel<'a, SimpleGridParameters<'a>> {
    pub fn new(parameter: &'a SimpleGridParameters) -> Result<Self> {
        let device = parameter.device;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        //let in_d: usize = (parameter.max_x * parameter.max_y * 2) as usize;
        //let out_d = 1_usize; // a simple score of log p
        //let ln1 = candle_nn::linear(in_d, 128, vb.pp("ln1"))?;
        //let ln2 = candle_nn::linear(128, out_d, vb.pp("ln2"))?;
        let f0 = vb
            .get_with_hints((1, 1), "f0", candle_nn::Init::Const(5.0))
            .unwrap();
        let hint = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.05,
        };
        // let hint = candle_nn::Init::default();
        let values = vb
            .get_with_hints(
                ((parameter.max_x * parameter.max_y) as usize, 1_usize),
                "values",
                hint,
            )
            .unwrap();

        let f = vb
            .get_with_hints(
                ((parameter.max_x * parameter.max_y) as usize, 1_usize),
                "f",
                hint,
            )
            .unwrap();
        //println!("all vars: {:?}", varmap.data());
        Ok(Self {
            //ln1,
            //ln2,
            f0,
            values,
            f,
            varmap,
            parameter,
        })
    }
}

impl<'a> ModelTrait<(u32, u32)> for SimpleGridModel<'a, SimpleGridParameters<'a>> {
    fn forward_ss_flow(
        &self,
        source: &impl StateTrait<(u32, u32)>,
        sink: &impl StateTrait<(u32, u32)>,
    ) -> Result<Tensor> {
        // let input1 = source.get_tensor()?;
        // let input2 = sink.get_tensor()?;
        // let inputs = Tensor::stack(&[&Tensor::cat(&[&input1, &input2], 0)?], 0)?.detach()?;
        // let xs = self.ln1.forward(&inputs)?.detach()?;
        // let xs = xs.relu()?;
        // let xs = self.ln2.forward(&xs)?.exp()?;
        let source_id = source.get_id();
        let sink_id = sink.get_id();
        let device = self.parameter.device;
        let source_id = Tensor::new(&[source_id], device)?;
        let sink_id = Tensor::new(&[sink_id], device)?;
        let sink_score = self.values.embedding(&sink_id)?;
        let flow = self.f.embedding(&source_id)?;

        let out_tensor = self
            .values
            .embedding(&source_id)?
            .sub(&sink_score)?
            .add(&flow)?
            .exp()?;
        //println!("DDD {} {}", ids, xs);
        Ok(out_tensor)
    }
    fn get_f0(&self) -> Result<&Tensor> {
        Ok(&self.f0)
    }

    fn forward_ss_flow_batch(
        &self,
        ss_pairs: &[(StateIdType, StateIdType)],
        batch_size: usize,
    ) -> Result<Tensor> {
        let mut source_ids = Vec::<u32>::new();
        let mut sink_ids = Vec::<u32>::new();
        ss_pairs.iter().for_each(|&(s, t)| {
            source_ids.push(s);
            sink_ids.push(t);
        });
        let device = self.parameter.device;

        let mut out_tensors = Vec::<_>::new();
        (0..ss_pairs.len())
            .step_by(batch_size)
            .try_for_each(|start_idx| -> Result<()> {
                let end_idx = if start_idx + batch_size < ss_pairs.len() {
                    start_idx + batch_size
                } else {
                    ss_pairs.len()
                };

                let batch_source_ids = Tensor::new(&source_ids[start_idx..end_idx], device)?;
                let batch_sink_ids = Tensor::new(&sink_ids[start_idx..end_idx], device)?;
                let sink_score = self.values.embedding(&batch_sink_ids)?;
                let flow = self.f.embedding(&batch_source_ids)?;
                let out_tensor = self
                    .values
                    .embedding(&batch_source_ids)?
                    .sub(&sink_score)?
                    .add(&flow)?
                    .exp()?;
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
        _source: &impl StateTrait<(u32, u32)>,
        _sink: &impl StateTrait<(u32, u32)>,
    ) -> Result<Tensor> {
        unimplemented!()
    }
}

pub type SimpleGridState = State<(u32, u32)>;

impl SimpleGridState {
    pub fn new(
        id: StateIdType,
        coordinate: (u32, u32),
        is_terminal: bool,
        reward: f32,
        parameter: &SimpleGridParameters,
    ) -> Self {
        let size = (parameter.max_x * parameter.max_y) as usize;
        let mut data = vec![0.0_f32; (parameter.max_x * parameter.max_y) as usize];
        data[(coordinate.0 * parameter.max_x + coordinate.1) as usize] = 1.0;
        let tensor = Tensor::from_vec(data, size, parameter.device).expect("create tensor fail");
        Self {
            id,
            data: coordinate,
            tensor,
            is_terminal,
            reward,
        }
    }

    pub fn set_reward(&mut self, reward: f32) {
        self.reward = reward;
    }
}

impl StateTrait<(u32, u32)> for SimpleGridState {
    fn get_id(&self) -> u32 {
        self.id
    }

    fn get_tensor(&self) -> Result<&Tensor> {
        Ok(&self.tensor)
    }

    fn get_forward_flow(
        &self,
        next_state: &impl StateTrait<(u32, u32)>,
        model: &impl ModelTrait<(u32, u32)>,
    ) -> Result<Tensor> {
        model.forward_ss_flow(self, next_state)
    }

    fn get_previous_flow(
        &self,
        previous_state: &impl StateTrait<(u32, u32)>,
        model: &impl ModelTrait<(u32, u32)>,
    ) -> Result<Tensor> {
        model.forward_ss_flow(previous_state, self)
    }

    fn get_data(&self) -> (u32, u32) {
        self.data
    }
}

pub type SimpleGridStateCollection = StateCollection<StateIdType, SimpleGridState>;

pub type SimpleGridMDP = MDP<StateIdType, SimpleGridState>;

impl<'a> MDPTrait<StateIdType, SimpleGridState, SimpleGridParameters<'a>> for SimpleGridMDP {
    fn mdp_next_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
        parameters: &SimpleGridParameters,
    ) -> Option<Vec<StateIdType>> {
        let state = collection.map.get(&state_id).expect("can get the stat");
        let coordinate = state.data;
        let max_x = parameters.max_x;
        let max_y = parameters.max_y;
        let mut next_delta: Vec<(u32, u32)> = Vec::new();
        let mut next_states = Vec::<StateIdType>::new();
        if coordinate.0 < max_x - 1 {
            next_delta.push((1, 0));
        }
        if coordinate.1 < max_y - 1 {
            next_delta.push((0, 1));
        }
        //println!("DBG: {:?} {:?} {} {}", coordinate, next_delta, parameters.max_x, parameters.max_y);
        next_delta.iter().for_each(|d| {
            let (x, y) = (coordinate.0 + d.0, coordinate.1 + d.1);
            let id = x * max_x + y;
            if collection.map.contains_key(&id) {
                next_states.push(id);
            } else {
                let new_state = Box::new(SimpleGridState::new(id, (x, y), false, 0.0, parameters));
                collection.map.entry(id).or_insert(new_state);
                next_states.push(id);
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
        collection: &mut SimpleGridStateCollection,
        parameters: &SimpleGridParameters,
    ) -> Option<StateIdType> {
        let mut rng = rand::thread_rng();
        self.mdp_next_possible_states(state_id, collection, parameters)
            .map(|states| states[rng.gen::<usize>().rem_euclid(states.len())])
    }

    fn mdp_previous_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
        parameters: &SimpleGridParameters,
    ) -> Option<Vec<StateIdType>> {
        let state = collection.map.get(&state_id).expect("can get the stat");
        let coordinate = state.data;
        let max_x = parameters.max_x;
        let mut next_delta: Vec<(u32, u32)> = Vec::new();
        let mut next_states = Vec::<StateIdType>::new();
        if coordinate.0 > 0 {
            next_delta.push((1, 0));
        }
        if coordinate.1 > 0 {
            next_delta.push((0, 1));
        }
        next_delta.iter().for_each(|d| {
            let (x, y) = (coordinate.0 - d.0, coordinate.1 - d.1);
            let id = x * max_x + y;
            if collection.map.contains_key(&id) {
                next_states.push(id);
            } else {
                let new_state = Box::new(SimpleGridState::new(id, (x, y), false, 0.0, parameters));
                collection.map.entry(id).or_insert(new_state);
                next_states.push(id);
            }
        });
        if next_states.is_empty() {
            None
        } else {
            Some(next_states)
        }
    }

    fn mdp_previous_one_uniform(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
        parameters: &SimpleGridParameters,
    ) -> Option<StateIdType> {
        let mut rng = rand::thread_rng();
        self.mdp_previous_possible_states(state_id, collection, parameters)
            .map(|states| states[rng.gen::<usize>().rem_euclid(states.len())])
    }
}

pub type SimpleGridSampler = Sampler<StateIdType>;
pub type SimpleGridSamplingConfiguration<'a> = SamplingConfiguration<
    'a,
    StateIdType,
    SimpleGridState,
    SimpleGridModel<'a, SimpleGridParameters<'a>>,
    SimpleGridParameters<'a>,
>;

impl<'a> Sampling<StateIdType, SimpleGridSamplingConfiguration<'a>> for SimpleGridSampler {
    fn sample_a_new_trajectory(
        &mut self,
        config: &mut SimpleGridSamplingConfiguration,
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

    fn sample_trajectories(&mut self, config: &mut SimpleGridSamplingConfiguration) {
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

    use candle_core::Tensor;

    use crate::{sampler::{MDPTrait, Sampling}, trajectory};

    use super::*;
    #[test]
    fn test_simple_grid_mdp() {
        use crate::simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).expect("no cuda device available")
        } else {
            Device::Cpu
        };
        let device = &device;
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000,
            rewards: vec![],
            device
        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0001,  parameters);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();

        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection, parameters) {
            println!("state_id: {:?}", s);
            state_id = s;
        }
    }

    #[test]
    fn generate_trajectory() {
        use crate::simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).expect("no cuda device available")
        } else {
            Device::Cpu
        };
        let device = &device;
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 200,
            rewards: vec![],
            device
        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0,  parameters);
        let collection = &mut SimpleGridStateCollection::default();
        let mut traj = trajectory::Trajectory::new();
        traj.push(state_id);
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();

        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection,  parameters) {
            println!("state_id: {:?}", s);
            state_id = s;
            traj.push(state_id);
        }
        println!("{:?}", traj.get_parent_offspring_pairs());
    }

    #[test]
    fn test_simple_grid_model() {
        use candle_core::Device;
        use crate::model::*;
        use crate::simple_grid_gfn::*;
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).expect("no cuda device available")
        } else {
            Device::Cpu
        };
        let device = &device;
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000,
            rewards: vec![],
            device
        };
        let s0 = SimpleGridState::new(0, (0, 0), false, 0.0,  parameters);
        let s1 = SimpleGridState::new(1, (1, 1), false, 0.0,  parameters);
        let model = SimpleGridModel::new( parameters).unwrap();
        let out = model.forward_ss_flow(&s0, &s1).unwrap();
        println!("{:?} {}", out, out);
    }

    #[test]
    fn test_simple_grid_model_with_a_trajectory() {
        use crate::model::ModelTrait;
        use candle_core::Device;
        use candle_nn::*;

        let c = (0u32, 0u32);

        let state_id = 0_u32;
        let device = if cfg!(feature = "cuda") {
            Device::new_cuda(0).expect("no cuda device available")
        } else {
            Device::Cpu
        };
        let device = &device;
        let parameters = &SimpleGridParameters {
            max_x: 12,
            max_y: 12,
            number_trajectories: 200,
            rewards: vec![ ((6, 7), 15.0),
            ((2, 5), 12.0),
            ((3, 9), 25.0),
            ((8, 2), 12.0),
            ((9, 2), 12.0),
            ((7, 2), 12.0),
            ((10, 5), 12.0),
            ((9, 5), 12.0),
            ((8, 5), 24.0),
            ((9, 8), 12.0),
            ((9, 7), 16.0),
            ((9, 6), 12.0),
            ((6, 11), 12.0)],
            device
        };
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, parameters);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));

        (0..parameters.max_x).for_each(|idx| {
            let state_id = idx * parameters.max_x + parameters.max_y - 1;
            let state: SimpleGridState = SimpleGridState::new(
                state_id,
                (idx, parameters.max_y - 1),
                true,
                0.001,
                parameters,
            );
            collection.map.insert(state_id, Box::new(state));
        });

        (0..parameters.max_y).for_each(|idx| {
            let state_id = (parameters.max_x - 1) * parameters.max_x + idx;
            let state: SimpleGridState = SimpleGridState::new(
                state_id,
                (parameters.max_x - 1, idx),
                true,
                0.001,
                parameters,
            );
            collection.map.insert(state_id, Box::new(state));
        });

        // let state_id = 2 * parameters.max_x + 12;
        // let state: SimpleGridState =
        //     SimpleGridState::new(state_id, (2, 12), true, 10.0, device, parameters);
        // collection.map.insert(state_id, Box::new(state));
 
        parameters.rewards.iter().for_each(|&((x, y), r)| {
            let state_id = x * parameters.max_x + y;
            let state: SimpleGridState =
                SimpleGridState::new(state_id, (x, y), true, r, parameters);
            collection.map.insert(state_id, Box::new(state));
        });



        let model = SimpleGridModel::new(parameters).unwrap();

        let mut sampler = SimpleGridSampler::new();

        let mdp = &mut SimpleGridMDP::new();

        let mut config = SimpleGridSamplingConfiguration {
            begin: 0,
            collection,
            mdp,
            model: Some(&model),
            device,
            parameters,
        };

        let mut opt = candle_nn::SGD::new(model.varmap.all_vars(), 0.0005).unwrap();

        (0..2).for_each(|i| {
            sampler.trajectories.clear();
            sampler.sample_trajectories(&mut config);

            (0..2).for_each(|j| {
                let mut losses = Vec::<_>::new();
                // let mut losses_sum = Vec::<_>::new();
                sampler.trajectories.iter().for_each(|traj| {
                    // println!("traj len: {}", traj.trajectory.len());

                    let trajectory = &traj.trajectory;

                    // println!("trajectory len: {}", trajectory.len());

                    trajectory.iter().for_each(|&state_id| {
                        let mut out_flow = Vec::<_>::new();
                        let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                        if s0.is_terminal {
                            let reward = s0.reward;

                            let reward = Tensor::from_slice(&[reward; 1], (1, 1), device).unwrap();
                            out_flow.push(reward);
                        }

                        if let Some(next_state_ids) = config.mdp.mdp_next_possible_states(
                            state_id,
                            config.collection,
                            parameters,
                        ) {
                            next_state_ids.into_iter().for_each(|next_state_id| {
                                // println!("DBG: {} {}", state_id, next_state_id);
                                let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                                let s1 =
                                    config.collection.map.get(&next_state_id).unwrap().as_ref();
                                let tmp = model.forward_ss_flow(s0, s1).unwrap();
                                // println!("outflow0: {:?} {:?} {}", s0, s1, tmp);
                                out_flow.push(tmp);
                            });
                        }
                        let tmp = Tensor::from_slice(&[0.001_f32; 1], (1, 1), device).unwrap();
                        out_flow.push(tmp);

                        // println!("outflow; {:?}", out_flow);
                        let out_flow = Tensor::stack(&out_flow[..], 0).unwrap();
                        let out_flow = out_flow.sum_all().unwrap();
                        // println!("out flow sum: {}", out_flow);
                        let out_flow = out_flow.log().unwrap();

                        let mut in_flow = Vec::<_>::new();
                        if state_id == 0 {
                            in_flow.push(model.get_f0().unwrap().clone());
                        }

                        if let Some(previous_state_ids) = config.mdp.mdp_previous_possible_states(
                            state_id,
                            config.collection,
                            parameters,
                        ) {
                            previous_state_ids
                                .into_iter()
                                .for_each(|previous_state_id| {
                                    let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                                    let s1 = config
                                        .collection
                                        .map
                                        .get(&previous_state_id)
                                        .unwrap()
                                        .as_ref();
                                    let tmp = model.forward_ss_flow(s1, s0).unwrap();
                                    // println!("inflow; {:?} {:?} {}", s0, s1, tmp);
                                    in_flow.push(tmp);
                                });
                        }
                        let tmp = Tensor::from_slice(&[0.001_f32; 1], (1, 1), device).unwrap();
                        in_flow.push(tmp);

                        let in_flow = Tensor::stack(&in_flow[..], 0).unwrap();
                        let in_flow = in_flow.sum_all().unwrap();
                        // println!("inflow sum; {}", in_flow);
                        let in_flow = in_flow.log().unwrap();

                        // let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                        // println!("in/out {:?} {}/{}", s0.data, out_flow, in_flow);

                        let loss_at_s = out_flow.sub(&in_flow).unwrap().sqr().unwrap();

                        losses.push(loss_at_s);
                    });
                    // println!("----");
                });

                let losses = Tensor::stack(&losses[..], 0).unwrap().sum_all().unwrap();
                let _ = opt.backward_step(&losses);
                println!(
                    "batch {} {}, total loss: {} {}",
                    i,
                    j,
                    losses,
                    model.get_f0().unwrap()
                );

                println!(
                    "{} {} # of trajectories: {}",
                    i,
                    j,
                    sampler.trajectories.len()
                );
            });
            println!("F0:{}", model.get_f0().unwrap());
            (0..parameters.max_x).for_each(|x| {
                (0..parameters.max_y).for_each(|y| {
                    if x < parameters.max_x - 1 {
                        let s0id = x * parameters.max_x + y;
                        let s1id = (x + 1) * parameters.max_x + y;

                        if !config.collection.map.contains_key(&s0id)
                            || !config.collection.map.contains_key(&s1id)
                        {
                            return;
                        }

                        let s0 = config.collection.map.get(&s0id).unwrap().as_ref();
                        let s1 = config.collection.map.get(&s1id).unwrap().as_ref();
                        let out0 = model.forward_ss_flow(s0, s1).unwrap();
                        // println!("out: {}", out0);
                        let out0: f32 = out0
                            .squeeze(0)
                            .unwrap()
                            .get(0)
                            .unwrap()
                            .to_scalar()
                            .unwrap();
                        println!("F\t{}\t{}\t{}\t{}\t{}\t{}", i, x, y, x + 1, y, out0);
                    }
                    if y < parameters.max_y - 1 {
                        let s0id = x * parameters.max_x + y;
                        let s1id = x * parameters.max_x + y + 1;

                        if !config.collection.map.contains_key(&s0id)
                            || !config.collection.map.contains_key(&s1id)
                        {
                            return;
                        }

                        let s0 = config.collection.map.get(&s0id).unwrap().as_ref();
                        let s1 = config.collection.map.get(&s1id).unwrap().as_ref();
                        let out0: f32 = model
                            .forward_ss_flow(s0, s1)
                            .unwrap()
                            .squeeze(0)
                            .unwrap()
                            .get(0)
                            .unwrap()
                            .to_scalar()
                            .unwrap();
                        println!("F\t{}\t{}\t{}\t{}\t{}\t{}", i, x, y, x, y + 1, out0);
                    }
                });
            });
        });
    }
}

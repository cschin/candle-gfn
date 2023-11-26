use crate::model::ModelTrait;
use crate::sampler::{MDPTrait, Sampler, Sampling, SamplingConfiguration, MDP};
use crate::state::{State, StateCollection, StateIdType, StateTrait};
use crate::trajectory::Trajectory;
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use candle_nn::{Linear, Module, VarBuilder, VarMap, ops::softmax_last_dim};
use fxhash::FxHashMap;
use rand::{self, Rng};

#[derive(Clone)]
pub struct SimpleGridParameters {
    pub max_x: u32,
    pub max_y: u32,
    pub number_trajectories: u32,
    pub terminate_states: Vec<(u32, u32)>,
    pub rewards: FxHashMap<(u32, u32), f32>,
}

pub struct SimpleGridModel<'a, P> {
    ln1: Linear,
    ln2: Linear,
    pub varmap: VarMap,
    pub device: &'a Device,
    parameter: P,
}

impl<'a> SimpleGridModel<'a, SimpleGridParameters> {
    pub fn new(device: &'a Device, parameter: &SimpleGridParameters) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let in_d: usize = (parameter.max_x * parameter.max_y * 2) as usize;
        let out_d = 2_usize; // a simple score of log p
        let ln1 = candle_nn::linear(in_d, 100, vb.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, out_d, vb.pp("ln2"))?;
        //println!("all vars: {:?}", varmap.data());
        Ok(Self {
            ln1,
            ln2,
            varmap,
            device,
            parameter: parameter.clone(),
        })
    }
}

impl<'a> ModelTrait<(u32, u32)> for SimpleGridModel<'a, SimpleGridParameters> {
    fn forward_ss_flow(
        &self,
        source: &impl StateTrait<(u32, u32)>,
        sink: &impl StateTrait<(u32, u32)>,
    ) -> Result<Tensor> {
        let input1 = source.get_tensor()?;
        let input2 = sink.get_tensor()?;
        let inputs = Tensor::stack(&[&Tensor::cat(&[&input1, &input2], 0)?], 0)?.detach()?;
        let xs = self.ln1.forward(&inputs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = ops::leaky_relu(&xs, 0.01)?;
        Ok(xs)
    }
    fn reverse_ss_flow(
        &self,
        source: &impl StateTrait<(u32, u32)>,
        sink: &impl StateTrait<(u32, u32)>,
    ) -> Result<Tensor> {
        todo!()
    }
}

pub type SimpleGridState = State<(u32, u32)>;

impl SimpleGridState {
    pub fn new(
        id: StateIdType,
        coordinate: (u32, u32),
        is_terminal: bool,
        reward: f32,
        device: &Device,
        parameter: &SimpleGridParameters,
    ) -> Self {
        let size = (parameter.max_x * parameter.max_y) as usize;
        let mut data = vec![0.0_f32; (parameter.max_x * parameter.max_y) as usize];
        data[(coordinate.0 * parameter.max_x + coordinate.1) as usize] = 1.0;
        let tensor = Tensor::from_vec(data, size, device).expect("create tensor fail");
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

impl MDPTrait<StateIdType, SimpleGridState, Device, SimpleGridParameters> for SimpleGridMDP {
    fn mdp_next_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
        device: &Device,
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
        next_delta.iter().for_each(|d| {
            let (x, y) = (coordinate.0 + d.0, coordinate.1 + d.1);
            let id = x * max_x + y;
            if collection.map.contains_key(&id) {
                next_states.push(id);
            } else {
                let new_state = Box::new(SimpleGridState::new(
                    id,
                    (x, y),
                    false,
                    0.0,
                    device,
                    parameters,
                ));
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
        device: &Device,
        parameters: &SimpleGridParameters,
    ) -> Option<StateIdType> {
        let mut rng = rand::thread_rng();
        self.mdp_next_possible_states(state_id, collection, device, parameters)
            .map(|states| states[rng.gen::<usize>().rem_euclid(states.len())])
    }

    fn mdp_previous_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
        device: &Device,
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
                let new_state = Box::new(SimpleGridState::new(
                    id,
                    (x, y),
                    false,
                    0.0,
                    device,
                    parameters,
                ));
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
        device: &Device,
        parameters: &SimpleGridParameters,
    ) -> Option<StateIdType> {
        let mut rng = rand::thread_rng();
        self.mdp_previous_possible_states(state_id, collection, device, parameters)
            .map(|states| states[rng.gen::<usize>().rem_euclid(states.len())])
    }
}

pub type SimpleGridSampler = Sampler<StateIdType>;
pub type SimpleGridSamplingConfiguration<'a> = SamplingConfiguration<
    'a,
    StateIdType,
    SimpleGridState,
    SimpleGridModel<'a, SimpleGridParameters>,
    SimpleGridParameters,
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
        let device = config.device;
        let parameters = config.parameters;

        let mut traj = Trajectory::new();
        let mut state_id = begin;
        let model = model.unwrap();
        let mut rng = rand::thread_rng();

        while let Some(next_states) =
            mdp.mdp_next_possible_states(state_id, collection, device, parameters)
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
            let inv_temp: f32= 0.1;
            let sump: f32 = state_and_flow.iter().map(|v| (-inv_temp * v.1).exp()).sum();
            let mut acc_sump = Vec::new();
            //let epsilon: f32 = 0.05;
            state_and_flow
                .iter()
                .for_each(|(id, p)| acc_sump.push((*id, (-inv_temp * p).exp() / sump )));
            let t = rng.gen::<f32>();
            let mut updated = false;
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
                if collection.map.get(&state_id).unwrap().as_ref().is_terminal {
                    // println!("terminal node: {}", state_id);
                    break
                }
            }
        }
        traj
    }

    fn sample_trajectories(&mut self, config: &mut SimpleGridSamplingConfiguration) {
        let number_trajectories = config.parameters.number_trajectories;
        (0..number_trajectories).for_each(|_| {
            let trajectory = self.sample_a_new_trajectory(config);
            trajectory
                .get_parent_offspring_pairs()
                .iter()
                .for_each(|&(s0id, s1id)| {
                    self.offsprings.entry(s0id).or_default().push(s1id);
                    self.parents.entry(s1id).or_default().push(s0id);
                });
            self.trajectories.push(trajectory);
        });
    }
}

use std::marker::PhantomData;

use crate::model::ModelTrait;
use crate::sampler::{MDPTrait, Sampler, Sampling, MDP};
use crate::state::{State, StateCollection, StateIdType, StateTrait};
use crate::trajectory::Trajectory;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use rand::{self, Rng};

pub struct SimpleGridModel<'a> {
    ln1: Linear,
    ln2: Linear,
    pub varmap: VarMap,
    pub vb: VarBuilder<'a>,
    pub device: Device,
}

impl<'a> SimpleGridModel<'a> {
    pub fn new() -> Result<Self> {
        let varmap = VarMap::new();
        let device = Device::new_cuda(0).expect("no cuda device available");
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let ln1 = candle_nn::linear(4, 100, vb.pp("ln1"))?;
        let ln2 = candle_nn::linear(100, 1, vb.pp("ln2"))?;
        Ok(Self {
            ln1,
            ln2,
            varmap,
            vb,
            device,
        })
    }
}

impl<'a> ModelTrait for SimpleGridModel<'a> {
    fn forward_ssp(&self, source: &impl StateTrait, sink: &impl StateTrait) -> Result<Tensor> {
        let input1 = source.get_tensor()?;
        let input2 = sink.get_tensor()?;
        let inputs = Tensor::stack(&[&Tensor::cat(&[&input1, &input2], 0)?], 0)?;
        let xs = self.ln1.forward(&inputs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        Ok(xs)
    }
    fn reverse_ssp(&self, source: &impl StateTrait, sink: &impl StateTrait) -> Result<Tensor> {
        let input1 = source.get_tensor()?;
        let input2 = sink.get_tensor()?;
        let inputs = Tensor::stack(&[&Tensor::cat(&[&input1, &input2], 0)?], 0)?;
        let xs = self.ln1.forward(&inputs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        Ok(xs)
    }
}

pub type SimpleGridState = State<(u32, u32)>;

impl SimpleGridState {
    pub fn new(id: StateIdType, coordinate: (u32, u32), is_terminal: bool, reward: f32) -> Self {
        let device = Device::new_cuda(0).expect("no cuda device available");
        let data = [coordinate.0 as f32, coordinate.1 as f32];
        let tensor = Tensor::new(&data, &device).expect("create tensor fail");
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

impl StateTrait for SimpleGridState {
    fn get_id(&self) -> u32 {
        self.id
    }

    fn get_tensor(&self) -> Result<&Tensor> {
        Ok(&self.tensor)
    }

    fn get_forward_score(
        &self,
        next_state: &impl StateTrait,
        model: &impl ModelTrait,
    ) -> Result<Tensor> {
        model.forward_ssp(self, next_state)
    }

    fn get_previous_score(
        &self,
        previous_state: &impl StateTrait,
        model: &impl ModelTrait,
    ) -> Result<Tensor> {
        model.forward_ssp(previous_state, self)
    }
}

pub type SimpleGridStateCollection = StateCollection<StateIdType, SimpleGridState>;

pub type SimpleGridMDP = MDP<StateIdType, SimpleGridState>;
impl SimpleGridMDP {
    pub fn new() -> Self {
        Self {
            phantom0: PhantomData,
            phantom1: PhantomData,
        }
    }
}

impl Default for SimpleGridMDP {
    fn default() -> Self {
        Self::new()
    }
}

impl MDPTrait<StateIdType, SimpleGridState> for SimpleGridMDP {
    fn mdp_next_possible_states(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
    ) -> Option<Vec<StateIdType>> {
        let state = collection.map.get(&state_id).expect("can get the stat");
        let coordinate = state.data;
        let max_x = 32u32;
        let max_y = 32u32;
        let mut next_delta: Vec<(u32, u32)> = Vec::new();
        let mut next_states = Vec::<StateIdType>::new();
        if coordinate.0 < max_x {
            next_delta.push((1, 0));
        }
        if coordinate.1 < max_y {
            next_delta.push((0, 1));
        }
        next_delta.iter().for_each(|d| {
            let (x, y) = (coordinate.0 + d.0, coordinate.1 + d.1);
            let id = x * max_x + y;
            if collection.map.contains_key(&id) {
                next_states.push(id);
            } else {
                let new_state = Box::new(SimpleGridState::new(id, (x, y), false, 0.0));
                collection.map.entry(id).or_insert(new_state);
                next_states.push(id);
            }
        });
        Some(next_states)
    }

    fn mdp_next_one_uniform(
        &self,
        state_id: StateIdType,
        collection: &mut SimpleGridStateCollection,
    ) -> Option<StateIdType> {
        let state = collection.map.get(&state_id).expect("can get the stat");
        let coordinate = state.data;
        let max_x = 32u32;
        let max_y = 32u32;
        let mut next_delta: Vec<(u32, u32)> = Vec::new();
        if coordinate.0 < max_x {
            next_delta.push((1, 0));
        }
        if coordinate.1 < max_y {
            next_delta.push((0, 1));
        }
        let next_coordinate = if next_delta.is_empty() {
            None
        } else {
            let mut rng = rand::thread_rng();
            let d = next_delta[rng.gen::<usize>().rem_euclid(next_delta.len())];
            Some((coordinate.0 + d.0, coordinate.1 + d.1))
        };
        if let Some((x, y)) = next_coordinate {
            let id = x * max_x + y;
            if !collection.map.contains_key(&id) {
                let new_state = Box::new(SimpleGridState::new(id, (x, y), false, 0.0));
                collection.map.entry(id).or_insert(new_state);
            }
            Some(id)
        } else {
            None
        }
    }
}

pub type SimpleGridSampler = Sampler<StateIdType>;

impl Sampling<StateIdType, SimpleGridState, SimpleGridModel<'_>> for SimpleGridSampler {
    fn sample_a_new_trajectory<'a>(
        &mut self,
        begin: StateIdType,
        collection: &mut SimpleGridStateCollection,
        mdp: &mut SimpleGridMDP,
        model: Option<&SimpleGridModel>,
    ) -> Trajectory<StateIdType> {
        let mut traj = Trajectory::new();
        let mut state_id = begin;
        let model = model.unwrap();
        let epsilon: f32 = 0.05;
        let mut rng = rand::thread_rng();
        while let Some(next_states) = mdp.mdp_next_possible_states(state_id, collection) {
            let scores = next_states
                .into_iter()
                .map(|sid| {
                    let s0 = collection.map.get(&state_id).unwrap().as_ref();
                    let s1 = collection.map.get(&sid).unwrap().as_ref();
                    let p = model
                        .forward_ssp(s0, s1)
                        .unwrap()
                        .neg()
                        .unwrap()
                        .exp()
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .get(0)
                        .unwrap();
                    //println!("{:?}", p);
                    (sid, p.to_scalar::<f32>().unwrap())
                })
                .collect::<Vec<_>>();
            if scores.is_empty() {
                break;
            };
            let sump: f32 = scores.iter().map(|v| v.1).sum();
            let mut acc_sump = Vec::new();
            scores
                .iter()
                .for_each(|(id, p)| acc_sump.push((*id, (1.0 - epsilon) * p / sump)));
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
                let d = scores[rng.gen::<usize>().rem_euclid(scores.len())];
                state_id = d.0;
                traj.push(state_id);
            }
        }

        // while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection) {
        //     // println!("state_id: {:?}", s);
        //     state_id = s;
        //     traj.push(state_id);
        // }

        traj
    }

    fn sample_trajectories(
        &mut self,
        begin: StateIdType,
        collection: &mut SimpleGridStateCollection,
        mdp: &mut SimpleGridMDP,
        model: Option<&SimpleGridModel>,
        number_trajectories: usize,
    ) {
        (0..number_trajectories).for_each(|_| {
            let trajectory = self.sample_a_new_trajectory(begin, collection, mdp, model);
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

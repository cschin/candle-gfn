use crate::model::ModelTrait;
use crate::sampler::{MDPTrait, Sampler, Sampling, MDP, SamplingConfiguration};
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
    pub device: &'a Device,
}

impl<'a> SimpleGridModel<'a> {
    pub fn new(device: &'a Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
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
        let input1 = source.get_tensor()?.detach().unwrap();
        let input2 = sink.get_tensor()?.detach().unwrap();
        let inputs = Tensor::stack(&[&Tensor::cat(&[&input1, &input2], 0)?], 0)?;
        let inputs = inputs.detach().unwrap();
        let xs = self.ln1.forward(&inputs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        Ok(xs)
    }
    fn reverse_ssp(&self, source: &impl StateTrait, sink: &impl StateTrait) -> Result<Tensor> {
        let input1 = source.get_tensor()?.detach().unwrap();
        let input2 = sink.get_tensor()?.detach().unwrap();
        let inputs = Tensor::stack(&[&Tensor::cat(&[&input1, &input2], 0)?], 0)?;
        let inputs = inputs.detach().unwrap();
        let xs = self.ln1.forward(&inputs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        Ok(xs)
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
    ) -> Self {
        let data = [coordinate.0 as f32, coordinate.1 as f32];
        let tensor = Tensor::new(&data, device).expect("create tensor fail");
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

pub struct SimpleGridParameters {
    pub max_x: u32,
    pub max_y: u32,
    pub number_trajectories: u32,
}

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
                let new_state = Box::new(SimpleGridState::new(id, (x, y), false, 0.0, device));
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
}

pub type SimpleGridSampler = Sampler<StateIdType>;
pub type SimpleGridSamplingConfiguration<'a> = SamplingConfiguration<'a, StateIdType, SimpleGridState, SimpleGridModel<'a>, SimpleGridParameters>;


impl<'a> Sampling<StateIdType, SimpleGridSamplingConfiguration<'a>>
    for SimpleGridSampler
{
    fn sample_a_new_trajectory(
        &mut self,
        config: &mut SimpleGridSamplingConfiguration
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
        let epsilon: f32 = 0.05;
        let mut rng = rand::thread_rng();

        while let Some(next_states) =
            mdp.mdp_next_possible_states(state_id, collection, device, parameters)
        {
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
        traj
    }

    fn sample_trajectories(
        &mut self,
        config: &mut SimpleGridSamplingConfiguration
    ) {
        let number_trajectories = config.parameters.number_trajectories;
        (0..number_trajectories).for_each(|_| {
            let trajectory =
                self.sample_a_new_trajectory(config);
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

use std::collections::HashMap;
use std::marker::PhantomData;

use candle_core::Device;

use crate::state::{StateCollection, StateIdType};
use crate::trajectory::Trajectory;

pub trait MDPTrait<I, S, D, P> {
    fn mdp_next_possible_states(
        &self,
        state_id: I,
        collection: &mut StateCollection<I, S>,
        device: &D,
        parameters: &P,
    ) -> Option<Vec<I>>;
    fn mdp_next_one_uniform(
        &self,
        state_id: I,
        collection: &mut StateCollection<I, S>,
        device: &D,
        parameters: &P,
    ) -> Option<I>;
    fn mdp_previous_possible_states(
        &self,
        state_id: I,
        collection: &mut StateCollection<I, S>,
        device: &D,
        parameters: &P,
    ) -> Option<Vec<I>>;
    fn mdp_previous_one_uniform(
        &self,
        state_id: I,
        collection: &mut StateCollection<I, S>,
        device: &D,
        parameters: &P,
    ) -> Option<I>;

}

pub struct MDP<I, S> {
    pub _state_id: PhantomData<I>,
    pub _state: PhantomData<S>,
}

impl<I, S> MDP<I, S> {
    pub fn new() -> Self {
        Self {
            _state_id: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<I, S> Default for MDP<I, S> {
    fn default() -> Self {
        Self::new()
    }
}
pub struct SamplingConfiguration<'a, I: Copy, S, M, P> {
    pub begin: StateIdType,
    pub collection: &'a mut StateCollection<I, S>,
    pub mdp: &'a mut MDP<I, S>,
    pub model: Option<&'a M>,
    pub device: &'a Device,
    pub parameters: &'a P,
}
pub trait Sampling<I:Copy, C> {
    fn sample_a_new_trajectory(
        &mut self,
        config: &mut C 

    ) -> Trajectory<I>;

    fn sample_trajectories(
        &mut self,
        config: &mut C
    );
}

pub struct Sampler<I: Copy> {
    pub trajectories: Vec<Trajectory<I>>,
    pub offsprings: HashMap<I, Vec<I>>,
    pub parents: HashMap<I, Vec<I>>,
}

impl<I: Copy> Sampler<I> {
    pub fn new() -> Self {
        let trajectories = Vec::<Trajectory<I>>::new();
        let offsprings = HashMap::<I, Vec<I>>::default();
        let parents = HashMap::<I, Vec<I>>::default();
        Self {
            trajectories,
            offsprings,
            parents,
        }
    }
}

impl<I: Copy> Default for Sampler<I> {
    fn default() -> Self {
        Self::new()
    }
}

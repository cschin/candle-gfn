use std::collections::HashMap;
use std::marker::PhantomData;

use candle_core::Device;

use crate::state::{StateCollection, StateIdType};
use crate::trajectory::Trajectory;

pub trait MDPTrait<I, S> {
    fn mdp_next_possible_states(
        &self,
        state_id: I,
        collection: &mut StateCollection<I, S>,
        device: &Device
    ) -> Option<Vec<I>>;
    fn mdp_next_one_uniform(
        &self,
        state_id: I,
        collection: &mut StateCollection<I, S>,
        device: &Device
    ) -> Option<I>;
}

pub struct MDP<I, S> {
    pub phantom0: PhantomData<I>,
    pub phantom1: PhantomData<S>,
}

pub trait Sampling<I: Copy, S, M> {
    fn sample_a_new_trajectory(
        &mut self,
        begin: StateIdType,
        collection: &mut StateCollection<I, S>,
        mdp: &mut MDP<I, S>,
        model: Option<&M>,
        device: &Device
    ) -> Trajectory<I>;

    fn sample_trajectories(
        &mut self,
        begin: StateIdType,
        collection: &mut StateCollection<I, S>,
        mdp: &mut MDP<I, S>,
        mode: Option<&M>,
        device: &Device,
        number_trajectories: usize,
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

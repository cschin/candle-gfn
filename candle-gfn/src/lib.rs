pub mod model;
pub mod sampler;
pub mod simple_grid_gfn;
pub mod state;
pub mod trajectory;

#[cfg(test)]
mod tests {

    use crate::sampler::{MDPTrait, Sampling};

    use super::*;
    #[test]
    fn test_simple_grid_mdp() {
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();
        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection) {
            println!("state_id: {:?}", s);
            state_id = s;
        }
    }

    #[test]
    fn generate_trajectory() {
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0);
        let collection = &mut SimpleGridStateCollection::default();
        let mut traj = trajectory::Trajectory::new();
        traj.push(state_id);
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();
        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection) {
            println!("state_id: {:?}", s);
            state_id = s;
            traj.push(state_id);
        }
        println!("{:?}", traj.get_parent_offspring_pairs());
    }

    #[test]
    fn test_simple_grid_model() {
        use model::*;
        use simple_grid_gfn::*;
        let s0 = SimpleGridState::new(0, (0, 0), false, 0.0);
        let s1 = SimpleGridState::new(1, (1, 1), false, 0.0);

        let model = SimpleGridModel::new().unwrap();
        let out = model.forward_ssp(&s0, &s1).unwrap();
        println!("{:?} {}", out, out);
    }

    #[test]
    fn test_simple_grid_model_with_a_trajectory() {
        use model::*;
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);

        let state_id = 0_u32;
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));

        let model = SimpleGridModel::new().unwrap();

        let mut sampler = SimpleGridSampler::new();
        let mut mdp = SimpleGridMDP::new();
        let traj = sampler.sample_a_new_trajectory(state_id, collection, &mut mdp, Some(&model));

        traj.get_parent_offspring_pairs()
            .iter()
            .for_each(|(s0id, s1id)| {
                let s0 = collection.map.get(s0id).unwrap().as_ref();
                let s1 = collection.map.get(s1id).unwrap().as_ref();
                let out = model.forward_ssp(s0, s1).unwrap();
                println!("{} {} {:?} {}", s0id, s1id, out, out)
            });
        let mut mdp = SimpleGridMDP::new();
        sampler.sample_trajectories(state_id, collection, &mut mdp, Some(&model), 10000);
    }
}

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
        use candle_core::Device;
        let device = Device::new_cuda(0).expect("no cuda device available");
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, &device);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 0,
        };
        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection, &device, parameters) {
            println!("state_id: {:?}", s);
            state_id = s;
        }
    }

    #[test]
    fn generate_trajectory() {
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        let device = Device::new_cuda(0).expect("no cuda device available");
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, &device);
        let collection = &mut SimpleGridStateCollection::default();
        let mut traj = trajectory::Trajectory::new();
        traj.push(state_id);
        collection.map.insert(state_id, Box::new(state));
        let mdp = SimpleGridMDP::new();
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 0,
        };
        while let Some(s) = mdp.mdp_next_one_uniform(state_id, collection, &device, parameters) {
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
        use candle_core::Device;
        let device = Device::new_cuda(0).expect("no cuda device available");
        let s0 = SimpleGridState::new(0, (0, 0), false, 0.0, &device);
        let s1 = SimpleGridState::new(1, (1, 1), false, 0.0, &device);

        let model = SimpleGridModel::new(&device).unwrap();
        let out = model.forward_ssp(&s0, &s1).unwrap();
        println!("{:?} {}", out, out);
    }

    #[test]
    fn test_simple_grid_model_with_a_trajectory() {
        use simple_grid_gfn::*;
        use candle_core::Device;
        use crate::model::ModelTrait;
        let c = (0u32, 0u32);

        let state_id = 0_u32;
        let device = &Device::new_cuda(0).expect("no cuda device available");
        let state: SimpleGridState = SimpleGridState::new(0, c, false, 0.0, device);
        let collection = &mut SimpleGridStateCollection::default();
        collection.map.insert(state_id, Box::new(state));

        let model = SimpleGridModel::new(device).unwrap();

        let mut sampler = SimpleGridSampler::new();
        let mdp = &mut SimpleGridMDP::new();
        let parameters = &SimpleGridParameters {
            max_x: 64,
            max_y: 64,
            number_trajectories: 1000
        };

        let mut config = SimpleGridSamplingConfiguration {
            begin: 0,
            collection, 
            mdp,
            model: Some(&model),
            device,
            parameters

        };

        let traj = sampler.sample_a_new_trajectory(&mut config);

        traj.get_parent_offspring_pairs()
            .iter_mut()
            .for_each(|(s0id, s1id)| {
                let s0 = config.collection.map.get(s0id).unwrap().as_ref();
                let s1 = config.collection.map.get(s1id).unwrap().as_ref();
                let out = model.forward_ssp(s0, s1).unwrap();
                println!("{} {} {:?} {}", s0id, s1id, out, out)
            });
        sampler.sample_trajectories(&mut config);
    }
}

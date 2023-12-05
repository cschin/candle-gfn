pub mod model;
pub mod sampler;
pub mod simple_grid_gfn;
pub mod state;
pub mod trajectory;

#[cfg(test)]
mod tests {

    use candle_core::Tensor;
    use fxhash::FxHashMap;

    use crate::sampler::{MDPTrait, Sampling};

    use super::*;
    #[test]
    fn test_simple_grid_mdp() {
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        //let device = &Device::new_cuda(0).expect("no cuda device available");
        let device = &Device::Cpu;
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
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);
        let mut state_id = 0_u32;
        use candle_core::Device;
        //let device = Device::new_cuda(0).expect("no cuda device available");
        let device = &Device::Cpu;
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
        use model::*;
        use simple_grid_gfn::*;
        //let device = Device::new_cuda(0).expect("no cuda device available");
        let device = &Device::Cpu;
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
        use simple_grid_gfn::*;
        let c = (0u32, 0u32);

        let state_id = 0_u32;
        //let device = &Device::new_cuda(0).expect("no cuda device available");
        let device = &Device::Cpu;
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
                            in_flow.push(model.get_f0().unwrap());
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

const VERSION_STRING: &str = env!("VERSION_STRING");
use candle_core::{shape, Device, Tensor};
use candle_gfn::{model::ModelTrait, sampler::*, simple_grid_gfn::*};
use candle_nn::*;
use clap::{self, CommandFactory, Parser};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{self, Path};

/// Align long contigs and identify potential SV regions with respect to the reference fasta file
#[derive(Parser, Debug)]
#[clap(name = "simple_2D_grid_gfn")]
#[clap(author, version)]
#[clap(about, long_about = None)]
struct CmdOptions {
    grid_spec_file: String,
    output_prefix: String,
    #[clap(long, short, default_value = None)]
    model_file: Option<String>,
    #[clap(long, short, default_value_t = 100)]
    batch_size: usize,
    #[clap(long, short, default_value_t = 100)]
    opt_cycles: usize,
    #[clap(long, short, default_value_t = 100)]
    number_of_batches: usize,
    #[clap(long, short, default_value_t = false)]
    save_all_batches: bool,
}

#[allow(dead_code)] // need the standard names for deserialization if they are not use
#[derive(Deserialize, Clone, Debug)]
struct GridSpec {
    max_x: u32,
    max_y: u32,
    rewards: Vec<((u32, u32), f32)>,
}

#[allow(dead_code)] // need the standard names for deserialization if they are not use
#[derive(Serialize, Clone, Debug)]
struct FlowSpec {
    f0: Vec<(usize, f32)>,
    loss: Vec<(usize, usize, f32)>,
    flow: Vec<(usize, (u32, u32), (u32, u32), f32)>,
}

#[allow(dead_code)] // need the standard names for deserialization if they are not use
#[derive(Serialize, Clone, Debug)]
struct TrajSpec {
    trajectories: Vec<(usize, Vec<(u32, u32)>)>,
}

fn main() -> Result<(), std::io::Error> {
    CmdOptions::command().version(VERSION_STRING).get_matches();
    let args = CmdOptions::parse();

    let spec_file = BufReader::new(File::open(Path::new(&args.grid_spec_file))?);
    let spec: GridSpec = serde_json::from_reader(spec_file)?;

    let state_id = 0_u32;
    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0).expect("no cuda device available")
    } else {
        Device::Cpu
    };
    let device = &device;
    let parameters = &SimpleGridParameters {
        max_x: spec.max_x,
        max_y: spec.max_y,
        number_trajectories: args.batch_size as u32,
        rewards: spec.rewards,
        device
    };
    let state: SimpleGridState = SimpleGridState::new(0, (0, 0), false, 0.0, parameters);
    let collection = &mut SimpleGridStateCollection::default();
    collection.map.insert(state_id, Box::new(state));

    (0..parameters.max_x).for_each(|idx| {
        (0..parameters.max_y).for_each(|idx2| {
            let state_id = idx * parameters.max_x + idx2;
            let state: SimpleGridState = SimpleGridState::new(
                state_id,
                (idx, idx2),
                true,
                0.1,
                parameters,
            );
            collection.map.insert(state_id, Box::new(state));
        });
    });


    parameters.rewards.iter().for_each(|&((x, y), r)| {
        let state_id = x * parameters.max_x + y;
        let state: SimpleGridState =
            SimpleGridState::new(state_id, (x, y), true, r, parameters);
        collection.map.insert(state_id, Box::new(state));
    });

    let mut model = SimpleGridModel::new(parameters).unwrap();

    if let Some(model_file) = args.model_file {
        println!("use model: {}", model_file);
        let _ = model.varmap.load(Path::new(&model_file));
    };

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

    let mut opt = candle_nn::SGD::new(model.varmap.all_vars(), 0.001).unwrap();

    let mut flow = FlowSpec {
        f0: vec![],
        loss: vec![],
        flow: vec![],
    };

    let mut out_traj = TrajSpec {
        trajectories: vec![],
    };

    (0..args.number_of_batches).for_each(|batch_idx| {
        sampler.trajectories.clear();
        sampler.sample_trajectories(&mut config);

        if args.save_all_batches || batch_idx == args.number_of_batches - 1 {
            sampler.trajectories.iter().for_each(|traj| {
                let mut trajectory_data: Vec<(u32, u32)> = Vec::new();
                let trajectory = &traj.trajectory;
                trajectory.iter().for_each(|&state_id| {
                    let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                    trajectory_data.push(s0.data);
                });
                out_traj.trajectories.push((batch_idx, trajectory_data));
            });
        };

        (0..args.opt_cycles).for_each(|cycle_idx| {
            let mut losses = Vec::<_>::new();
            sampler.trajectories.iter().for_each(|traj| {
                let trajectory = &traj.trajectory;

                trajectory.iter().for_each(|&state_id| {
                    let mut out_flow = Vec::<_>::new();
                    let s0 = config.collection.map.get(&state_id).unwrap().as_ref();

                    if s0.is_terminal {
                        let reward = s0.reward;

                        let reward = Tensor::new(&[[reward; 1]], device).unwrap();
                        out_flow.push(reward);
                    }

                    if let Some(next_state_ids) = config.mdp.mdp_next_possible_states(
                        state_id,
                        config.collection,
                        parameters,
                    ) {
                        next_state_ids.into_iter().for_each(|next_state_id| {
                            let s0 = config.collection.map.get(&state_id).unwrap().as_ref();
                            let s1 = config.collection.map.get(&next_state_id).unwrap().as_ref();
                            let tmp = model.forward_ss_flow(s0, s1).unwrap();
                            out_flow.push(tmp);
                        });
                    }
                    let tmp = Tensor::from_slice(&[0.001_f32; 1], (1, 1), device).unwrap();
                    out_flow.push(tmp);

                    let out_flow = Tensor::stack(&out_flow[..], 0).unwrap();
                    let out_flow = out_flow.sum_all().unwrap();
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
                                in_flow.push(tmp);
                            });
                    }
                    let tmp = Tensor::new(&[[0.001_f32; 1]], device).unwrap();
                    in_flow.push(tmp);

                    let in_flow = Tensor::stack(&in_flow[..], 0).unwrap();
                    let in_flow = in_flow.sum_all().unwrap();
                    let in_flow = in_flow.log().unwrap();
                    let loss_at_s = out_flow.sub(&in_flow).unwrap().sqr().unwrap();

                    losses.push(loss_at_s);
                });
            });

            let losses = Tensor::stack(&losses[..], 0).unwrap().sum_all().unwrap();
            let _ = opt.backward_step(&losses);
            let loss_scalar: f32 = losses.reshape(shape::SCALAR).unwrap().to_scalar().unwrap();
            let f0_scalar: f32 = model
                .get_f0()
                .unwrap()
                .reshape(shape::SCALAR)
                .unwrap()
                .to_scalar()
                .unwrap();
            println!(
                "batch {} {}, total loss: {} {}",
                batch_idx, cycle_idx, loss_scalar, f0_scalar
            );
            flow.loss.push((batch_idx, cycle_idx, loss_scalar));
        });

        if args.save_all_batches || batch_idx == args.number_of_batches - 1 {
            let f0: f32 = model
                .get_f0()
                .unwrap()
                .flatten_all()
                .unwrap()
                .get(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            flow.f0.push((batch_idx, f0));

            //println!("F0:{}",f0);
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
                        let out0: f32 = out0
                            .squeeze(0)
                            .unwrap()
                            .get(0)
                            .unwrap()
                            .to_scalar()
                            .unwrap();
                        flow.flow.push((batch_idx, (x, y), (x + 1, y), out0));
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
                        flow.flow.push((batch_idx, (x, y), (x, y + 1), out0));
                    }
                });
            });
        };
        let mut flow_file = BufWriter::new(
            File::create(path::Path::new(&args.output_prefix).with_extension("flow"))
                .expect("can't create the output flow file"),
        );

        let flow = serde_json::to_string(&flow).unwrap();
        let _ = flow_file.write_all(flow.as_bytes());

        let mut traj_file = BufWriter::new(
            File::create(path::Path::new(&args.output_prefix).with_extension("traj"))
                .expect("can't create the output traj file"),
        );

        let out_traj = serde_json::to_string(&out_traj).unwrap();
        let _ = traj_file.write_all(out_traj.as_bytes());

        //println!("varmap: {:?}", config.model.unwrap().varmap.all_vars());
        let _ = config
            .model
            .unwrap()
            .varmap
            .save(path::Path::new(&args.output_prefix).with_extension("safetensors"));
    });
    Ok(())
}

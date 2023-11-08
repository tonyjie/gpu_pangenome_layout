'''
PyTorch implementation of pangenome graph layout. 

'''


import time
import argparse
import torch
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from odgi_dataset import OdgiInterface, OdgiDataloader
import torch.profiler as profiler

# True: use gradient-based method; False: use vector implementation
GRADIENT_METHOD = False
PRINT_LOG = False
PROFILE = True
PYTORCH_PROFILE = False
NSYS_PROFILE = False

# default parameters
zipf_theta=0.99
space_max=1000
space_quantization_step=100
# space = max_path_step_count

# DIR
DATASET_DIR="./dataset_array"

class PlaceEngine(nn.Module):
    '''
    @brief: Graph 2D layout Engine. It contains the parameters ([X, Y] coordinates of all nodes) to be updated. 
    '''
    def __init__(self, init_layout, num_nodes, lr_schedule, device):
        '''
        @brief initialization
        @param num_nodes: number of nodes in the graph
        '''
        super().__init__()
        self.lr_schedule = lr_schedule
        self.num_nodes = num_nodes
        self.pos = nn.Parameter(data=init_layout, requires_grad=True)
    
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1) # we'll set customized learning rate for each epoch
        
    def stress_fn(self, i, j, vis_p_i, vis_p_j, dis, iter):
        '''
        @brief Compute the stress function for batch_size pairs of nodes. This strictly follows the ODGI implementation. 
        '''        
        diff = self.pos[i] - self.pos[j]

        mag = torch.norm(diff, dim=1)
        # make sure no element in mag is 0
        mag = torch.max(mag, torch.ones_like(mag)*1e-9) # avoid mag = 0, will cause NaN
        coeff = 1 / (4 * torch.max(dis, torch.tensor(self.lr_schedule[iter])))
        stress = coeff * (mag - dis) ** 2
        # sum up the stress for each node
        stress_sum = torch.sum(stress, dim=0)
        # print(f"stress: {stress_sum}")
        return stress_sum

    def gradient_step(self, i, j, vis_p_i, vis_p_j, dis, iter):
        self.optimizer.zero_grad()


        stress = self.stress_fn(i, j, vis_p_i, vis_p_j, dis, iter)
        stress.backward()

        # customized learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule[iter]
        self.optimizer.step()

        if (torch.isnan(self.pos).any()):
            # breakpoint()
            raise ValueError(f"nan found in pos: {self.pos}\n stress: {stress}\n dis: {dis} \n i: {i} \n j: {j} \n vis_p_i: {vis_p_i} \n vis_p_j: {vis_p_j} \n iter: {iter}")

        return stress



def draw(pos_changes, output_dir):
    def draw_one_graph(pos, idx, output_dir):
        fig, ax = plt.subplots()

        xmin = pos.min()
        xmax = pos.max()
        edge = 0.1 * (xmax - xmin)
        ax.set_xlim(xmin-edge, xmax+edge)
        ax.set_ylim(xmin-edge, xmax+edge)
        ax.set_title(f"Iter {idx}")

        pos = pos.reshape((pos.shape[0]//2, 2, 2))

        for p in pos:
            plt.plot(p[:,0], p[:,1], '-', linewidth=1)
        plt.savefig(f"{output_dir}/iter{idx}.png")
        plt.close(fig)
        frames.append(imageio.v2.imread(f"{output_dir}/iter{idx}.png"))

    frames = list()
    for idx, pos in enumerate(pos_changes):
        draw_one_graph(pos, idx, output_dir)

    imageio.mimsave(f"{output_dir}/iter_animate.gif", frames, duration=0.1)


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if (device == "cpu"):
        raise ValueError("CPU is not good. Please use GPU.")
    print(f"==== Device: {device}; Chromosome: {args.input_chrom} ====")

    OG_FILE = f"{DATASET_DIR}/{args.input_chrom}.og"
    ARRAY_DIR = f"{DATASET_DIR}/{args.input_chrom}"
    RESULT_DIR = f"{ARRAY_DIR}/batch_size={args.batch_size}"

    def load_batch(batch_size, num_path, iter):
        '''
        Load batch data
        '''
        def zipf_batch(n, theta, zeta2, zetan):
            '''
            @brief
                zipf distribution, one kind of power law distirbution. Should return a torch tensor of shape [batch_size]
            @param
                n: tensor
                zetan: tensor
            @return
                val: tensor
            '''
            alpha = 1.0 / (1.0 - theta)
            denominator = 1.0 - zeta2 / zetan # tensor
            denominator = torch.where(denominator == 0, 1e-9, denominator)
            eta = (1.0 - torch.pow(2.0 / n, 1.0 - theta)) / denominator # tensor

            u = 1.0 - torch.rand(size=[batch_size], device=device) # tensor
            uz = u * zetan # tensor
            val = torch.where(uz < 1.0, 1, 0) # tensor
            val = torch.where((uz >= 1.0) & (uz < (1.0 + pow(0.5, zipf_theta))), 2, val)
            val = torch.where((uz >= (1.0 + pow(0.5, zipf_theta))), (1 + n * torch.pow(eta * u - eta + 1.0, alpha)).int(), val.int())
            val = torch.where(val > n, val - 1, val)
            # assert that no value in val >= 0 and <= n
            assert(torch.all(val <= n))
            assert(torch.all(val >= 0))
            return val.int()


        def backward(step_i):
            '''
            In the cooling stage, we go backward from step_i to pick step_j. 
            @Param: step_i, shape: [batch_size]
            @Return: step_j
            '''
            jump_space = step_i
            space = jump_space
            mask_cond = (jump_space > space_max)
            space = torch.where(mask_cond, space_max + torch.div((jump_space - space_max), space_quantization_step, rounding_mode='trunc') + 1, space)
            z = zipf_batch(jump_space, zipf_theta, zetas[2], zetas[space.long()])
            step_j = step_i - z
            return step_j

        def forward(step_i):
            '''
            In the cooling stage, we go forward from step_i to pick step_j
            @Param: step_i, shape: [batch_size]
            @Return: step_j
            '''
            jump_space = num_nodes_in_path[path_idx] - step_i - 1
            space = jump_space
            mask_cond = (jump_space > space_max)
            space = torch.where(mask_cond, space_max + torch.div((jump_space - space_max), space_quantization_step, rounding_mode='trunc') + 1, space)
            z = zipf_batch(jump_space, zipf_theta, zetas[2], zetas[space.long()])
            step_j = step_i + z
            return step_j

        def cooling(step_i):
            '''
            @Brief: cooling stage. Get the step_j given step_i. 
            @Param: step_i, shape: [batch_size]
            @Return: step_j, shape: [batch_size]
            '''
            # mask = True: backward; mask = False: forward
            mask = ((step_i > 0) & (torch.rand_like(step_i.float()) <= 0.5)) | (step_i == num_nodes_in_path[path_idx] - 1)
            step_j = torch.where(mask, backward(step_i), forward(step_i))
            return step_j
            
        def noncooling(step_i):
            '''
            @Brief: noncooling stage. Get the step_j given step_i. 
            @Param: step_i, shape: [batch_size]
            @Return: step_j, shape: [batch_size]
            '''
            step_j = (torch.rand(batch_size, device=device) * num_nodes_in_path[path_idx]).int()
            step_j = torch.where(step_j == step_i, ((step_j + 1) % num_nodes_in_path[path_idx]).int(), step_j)
            return step_j


        path_idx = torch.randint(0, num_path, (batch_size,), device=device)

        step_i = (torch.rand(batch_size, device=device) * num_nodes_in_path[path_idx]).int()

        cond_cooling = (iter >= 15) | (torch.rand(batch_size, device=device) <= 0.5) # condition for cooling stage
        step_j = torch.where(cond_cooling, cooling(step_i), noncooling(step_i))

        node_vis_i = torch.randint(0, 2, (batch_size,), device=device) # flipcoin on {0,1}
        node_vis_j = torch.randint(0, 2, (batch_size,), device=device) # flipcoin on {0,1}
        
        # 1d array for pos_ref, node_id
        dist_ref = torch.abs(pos_ref[start_step_idx[path_idx] + step_i*2 + node_vis_i] - pos_ref[start_step_idx[path_idx] + step_j*2 + node_vis_j])
        node_i = node_id[start_step_idx[path_idx] + step_i*2 + node_vis_i].long()
        node_j = node_id[start_step_idx[path_idx] + step_j*2 + node_vis_j].long()

        return node_i, node_j, node_vis_i, node_vis_j, dist_ref


    # ========== Load Precomputed Arrays ============
    # ===== read pos_ref: line by line =====
    num_path = 0
    pos_ref_list = list()

    with open(f"{ARRAY_DIR}/pos.txt", "r") as f:
        num_path = int(f.readline().strip())

        num_nodes_in_path = torch.zeros(num_path, dtype=torch.int32)
        start_step_idx = torch.zeros(num_path, dtype=torch.int32) # record the start step idx for each path

        for i in range(num_path):
            num_pangenome_nodes = int(f.readline().strip())
            num_nodes_in_path[i] = num_pangenome_nodes
            start_step_idx[i] = start_step_idx[i-1] + num_nodes_in_path[i-1] * 2 if i > 0 else 0

            arr = np.array(list(map(int, f.readline().strip().split())))
            pos_ref_list.append(arr)
            if (i % 100 == 0 and i > 0):
                print(f"[Position] path {i}: finish reading")

    max_nodes_in_path = torch.max(num_nodes_in_path)
    total_nodes_in_path = torch.sum(num_nodes_in_path)
    
    print(f"max_nodes_in_path: {max_nodes_in_path}")
    print(f"total_nodes_in_path: {total_nodes_in_path}")

    pos_ref = torch.zeros((total_nodes_in_path * 2), dtype=torch.int32)

    for i in range(num_path):
        pos_ref[start_step_idx[i]:start_step_idx[i] + num_nodes_in_path[i]*2] = torch.tensor(pos_ref_list[i], dtype=torch.int32)

    pos_ref = pos_ref.to(device)
    print(f"[Position] Finished Loading")

    # ===== read node_id: line by line =====
    node_id = torch.zeros((total_nodes_in_path * 2), dtype=torch.int32)

    # print(f"start_step_idx: {start_step_idx}")
    # print(f"num_nodes_in_path: {num_nodes_in_path}")

    with open(f"{ARRAY_DIR}/vis_id.txt", "r") as f:
        for i in range(num_path):
            line = f.readline().strip()
            numbers = list(map(int, line.split()))
            arr = np.array(numbers)
            node_id[start_step_idx[i]:start_step_idx[i] + num_nodes_in_path[i]*2] = torch.tensor(arr, dtype=torch.int32)
            if (i % 100 == 0 and i > 0):
                print(f"[Node ID] path {i}: finish reading")

    node_id = node_id.to(device)
    num_nodes_in_path = num_nodes_in_path.to(device)
    start_step_idx = start_step_idx.to(device)
    print(f"[Node ID] Finished Loading")

    # ===== read schedule from config.txt =====
    schedule = []
    with open(f"{ARRAY_DIR}/config.txt", "r") as f:
        num_pangenome_nodes = int(f.readline().strip())
        line = f.readline().strip()
        numbers = list(map(float, line.split()))
        schedule.extend(numbers)
        # we delete the last one, which is the 31st iteration (shouldn't use it)
        # schedule.pop()
    
    # print(f"==== Schedule: {schedule} ====")
    print(f"====== Finish Loading Schedule ======")

    steps_per_iteration = 10 * total_nodes_in_path
    num_nodes = 2 * num_pangenome_nodes
    # ===== read init_layout: line by line =====
    init_layout = torch.zeros((num_nodes, 2), dtype=torch.float32)
    with open(f"{ARRAY_DIR}/init_layout.txt", "r") as f:
        line_x = f.readline().strip()
        layout_x = np.array(list(map(float, line_x.split())))
        line_y = f.readline().strip()
        layout_y = np.array(list(map(float, line_y.split())))
        init_layout[:, 0] = torch.tensor(layout_x, dtype=torch.float32)
        init_layout[:, 1] = torch.tensor(layout_y, dtype=torch.float32)
    print(f"==== Finish Loading Init Layout ====") # [num_nodes, 2]

    # ======= compute zipf zetas =======
    space = max_nodes_in_path
    zetas_cnt = 0
    if space <= space_max:
        zetas_cnt = space
    else:
        zetas_cnt = space_max + (space - space_max) // space_quantization_step + 1
    zetas_cnt += 1
    print(f"zetas_cnt: {zetas_cnt}")

    zetas = torch.zeros(size=[zetas_cnt], dtype=torch.double)
    zeta_tmp = 0.0
    for i in range(1, space + 1):
        zeta_tmp += pow(1.0 / i, zipf_theta)
        if (i <= space_max):
            zetas[i] = zeta_tmp
        if (i >= space_max and (i - space_max) % space_quantization_step == 0):
            zetas[space_max + 1 + (i - space_max) // space_quantization_step] = zeta_tmp

    zetas = zetas.to(device)
    print(f"==== Finish Computing Zetas ====")


    print(f"==== num_pangenome_nodes: {num_pangenome_nodes}; steps_per_iteration: {steps_per_iteration} ====")


    # set pos on device, requires_grad=True
    pos = torch.tensor(init_layout, dtype=torch.float32, requires_grad=True, device=device)

    # ========= Gradient-based Method =========
    if GRADIENT_METHOD:
        mod = PlaceEngine(init_layout, num_nodes, schedule, device)
        mod = mod.to(device)


    pos_changes = np.zeros((args.num_iter, num_nodes, 2), dtype=np.float32)

    start = time.time()

    if PROFILE: 
        dataload_total = 0
        transfer_total = 0
        compute_total = 0

    # with profiler.profile(
    #     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], 
    #                 with_stack=False, profile_memory=False, record_shapes=False) as prof:
        # enable Pytorch profiler

            
    for iter in range(args.num_iter):
        eta = schedule[iter]

        if NSYS_PROFILE:
            if iter >= 1: 
                torch.cuda.cudart().cudaProfilerStart()

        if PROFILE:
            transfer_iter = 0
            compute_iter = 0
            dataload_iter = 0
            dataload_start = time.time()

        # for batch_idx, (i, j, vis_p_i, vis_p_j, _w, dis) in enumerate(data): # (i,j) start from 1; vis_p_i, vis_p_j is in {0,1}
        for batch_idx in range(steps_per_iteration // args.batch_size):
            i, j, vis_p_i, vis_p_j, dis = load_batch(args.batch_size, num_path, iter)


            if PROFILE:
                dataload_iter += time.time() - dataload_start
                if (device == torch.device("cuda")):
                    torch.cuda.synchronize()
                transfer_start = time.time()

            if PROFILE:
                if (device == torch.device("cuda")):
                    torch.cuda.synchronize()
                transfer_iter += time.time() - transfer_start

                compute_start = time.time()

            # ======== Wrap up within Pytorch nn.Module =======
            if GRADIENT_METHOD:
                stress_sum = mod.gradient_step(i, j, vis_p_i, vis_p_j, dis, iter)


            # ========= Gradient Update Implementation ========
            # diff = pos[i] - pos[j]
            # mag = torch.norm(diff, dim=1)
            # mag = torch.max(mag, torch.ones_like(mag)*1e-9) # avoid mag = 0, will cause NaN
            # coeff = 1 / (4 * torch.max(dis, torch.tensor(eta)))
            # stress = coeff * (mag - dis) ** 2
            # # sum up the stress for each node
            # stress_sum = torch.sum(stress, dim=0)
            # stress_sum.backward()
            # pos.data.sub_(eta * pos.grad.data)
            # pos.grad.data.zero_()

            # if (torch.isnan(pos).any()):
            #     # breakpoint()
            #     raise ValueError(f"nan found in pos: {pos}\n stress: {stress}\n dis: {dis} \n i: {i} \n j: {j} \n vis_p_i: {vis_p_i} \n vis_p_j: {vis_p_j} \n iter: {iter}")



            # ======== Vector Implementation ==========
            else:
                with torch.no_grad():
                    w = 1 / dis # torch.clamp in-place. 
                    mu = torch.min(w * eta, torch.ones_like(w, device=device)) # torch.ones_like() move out. Shape is consistent. 
                    diff = pos[i] - pos[j] # maybe cpu -> gpu for those index is time-consuming?
                    mag = torch.norm(diff, dim=1)
                    mag = torch.max(mag, torch.ones_like(mag, device=device)*1e-9) # avoid mag = 0, will cause NaN
                    r = (dis - mag) / 2
                    update = torch.unsqueeze(mu * r / mag, dim=1) * diff
                    pos[i] += update
                    pos[j] -= update # memory contention?  

                    if (torch.isnan(pos).any()):
                        # breakpoint()
                        raise ValueError(f"nan found in pos: {pos}\n dis: {dis} \n i: {i} \n j: {j} \n iter: {iter}")

            if PROFILE:
                if (device == torch.device("cuda")):
                    torch.cuda.synchronize()
                compute_iter += time.time() - compute_start

            if PRINT_LOG:
                if batch_idx % args.log_interval == 0:
                    # print(f"Iteration[{iter}]: {batch_idx}/{data.steps_in_iteration() // args.batch_size}. Stress: {stress_sum.item():.2f}")
                    print(f"Iteration[{iter}]: {batch_idx}/{steps_per_iteration // args.batch_size}")

            if PROFILE:
                dataload_start = time.time()

        if PROFILE:
            dataload_total += dataload_iter
            transfer_total += transfer_iter
            compute_total += compute_iter

            print(f"====== Time breakdown for Iter[{iter}] dataload: {dataload_iter:.2e}, transfer: {transfer_iter:.2e}, compute: {compute_iter:.2e} =====")

            if GRADIENT_METHOD:
                pos_changes[iter] = mod.pos.cpu().detach().numpy()
            else:
                pos_changes[iter] = pos.cpu().detach().numpy()



    elapsed = time.time() - start
    print(f"==== Elapsed time: {elapsed:.2f}s ====")
    if PROFILE:
        print(f"====== Time breakdown:  dataload: {dataload_total:.2e}, transfer: {transfer_total:.2e}, compute: {compute_total:.2e} =====")

    # print("==== self_cpu_time_total ====")
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    # print("==== cpu_time_total ====")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print("==== self_cuda_time_total ====")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    # print("==== cuda_time_total ====")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    

    
    if args.lay or args.draw:
        if not os.path.exists(RESULT_DIR):
            os.mkdir(RESULT_DIR)

    # generate ODGI .lay file
    if args.lay:
        for idx, pos in enumerate(pos_changes):
            pos_reshape = pos.reshape(num_nodes//2,2,2)
            OdgiInterface.generate_layout_file(odgi_load_graph(OG_FILE), pos_reshape, f"{RESULT_DIR}/iter{idx}.lay")

    if args.draw:
        draw(pos_changes, RESULT_DIR)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ODGI Vector Processing for Pairwise Update")
    parser.add_argument('input_chrom', type=str, default='DRB1-3123', help='odgi variation graph')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_iter', type=int, default=30, help='number of iterations')
    # parser.add_argument("--steps_per_iter", type=int, default=5, help="steps per iteration")
    parser.add_argument('--draw', action='store_true', default=False, help='draw the graph')
    parser.add_argument('--lay', action='store_true', default=False, help='generate .lay file for ODGI to draw')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    args = parser.parse_args()
    print(args)
    main(args)
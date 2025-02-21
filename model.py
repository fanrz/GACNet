import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

# get the distance between source node and destination nodes
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    print('in the square_distance function')
    B, N, _ = src.shape
    print('src.shape is ',src.shape)
    _, M, _ = dst.shape
    print('dst.shape is ',dst.shape)
    # Here, src is [24, 64, 3]
    # dst is [24, 16, 3]
    # in this step, dist is torch.Size([24, 64, 16])
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # in this step, dist is torch.Size([24, 64, 16])
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # in this step, dist is torch.Size([24, 64, 16])
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # you will get dist: torch.Size([24, 16, 64])
    print('dist.shape is ',dist.shape)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # some operation same as downsampling
    print('farthest_point_sample begin')
    device = xyz.device
    # B, N, C are torch.Size([24, 4096, 3])
    B, N, C = xyz.shape
    # centroids = torch.zeros(B, npoint) = torch.zeros(24, 1024) 
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # torch.Size([24, 1024])
    print('centroids.shape is', centroids.shape)
    # distance = torch.ones(B, N) = torch.zeros(24, 4096)
    # And why * 1e10 ????
    # distance torch.Size([24, 4096])
    distance = torch.ones(B, N).to(device) * 1e10
    print('distance.shape is', distance.shape)
    print(distance)
    # farthest = torch.randint(0, N, B)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    print('farthest.shape is')
    print(farthest.shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    print('batch_indices.shape is')
    print(batch_indices.shape)
    print('npoint is ')
    print(npoint) 
    print('\n') 
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        if i == 2:
            print('centroid.shape is')
            # torch.Size([24, 1, 3])
            print(centroid.shape)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: 1024
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    # this function
    print('in this function, sample_and_group')
    print('xyz.shape is',xyz.shape)
    print('points.shape is',points.shape)
    # npoint, radius, nsample: 1024, 0.1, 32
    print('npoint, radius, nsample are ',npoint, radius, nsample)
    # B, N, C is 24, 4096, 3
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    print('fps_idx is',fps_idx.shape)
    print(fps_idx)
    print('\n')
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class GraphAttention(nn.Module):
    def __init__(self,all_channel,feature_dim,dropout,alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p,delta_h],dim = -1) # [B, npoint, nsample, C+D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature),dim = 2) # [B, npoint, D]
        return graph_pooling


class GraphAttentionConvLayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all,droupout=0.6,alpha=0.2):
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout
        self.alpha = alpha
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.GAT = GraphAttention(3+last_channel,last_channel,self.droupout,self.alpha)
    
    def forward(self, xyz, points):
        """
        Input:
            # Here, [B, C, N] is [24, 3, 4096], B is batch, C is channel, N is num of Nodes. 
            xyz: input points position data, [B, C, N]
            # Here, [B, C, N] is [24, 6, 4096], B is batch, D is data of node? N is num of Nodes. 
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # [B, C, N] -> [B, N, C]
        # torch.Size([24, 4096, 3])
        xyz = xyz.permute(0, 2, 1)
        print('xyz.shape is', xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # choose this option
            new_xyz, new_points, grouped_xyz, fps_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, True)
            print('after sample_and_group')
            print('new_xyz is ',new_xyz)
            print('new_points is ',new_points)
            print('grouped_xyz is ',grouped_xyz)
            print('fps_points is ',fps_points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # fps_points: [B, npoint, C+D,1]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        print('shape of new_points is ',new_points.shape)
        print('shape of fps_points is ',fps_points.shape)
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1) # [B, C+D, 1,npoint]
        print('shape of fps_points is ',fps_points.shape)
        print('self.mlp_convs is', self.mlp_convs)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points =  F.relu(bn(conv(new_points)))
        # new_points: [B, F, nsample,npoint]
        # fps_points: [B, F, 1,npoint]
        new_points = self.GAT(center_xyz=new_xyz,
                              center_feature=fps_points.squeeze().permute(0,2,1),
                              grouped_xyz=grouped_xyz,
                              grouped_feature=new_points.permute(0,3,2,1))
        print('shape of new_points is ',new_points.shape)
        new_xyz = new_xyz.permute(0, 2, 1)
        print('shape of new_xyz is ',new_xyz.shape)
        new_points = new_points.permute(0, 2, 1)
        print('shape of new_points is ',new_points.shape)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    # l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
    # 
    # 
    # 
    # 
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        print('xyz1.shape is ',xyz1.shape)
        print('xyz2.shape is ',xyz2.shape)
        # print('points1.shape is ',points1.size())
        # print('points2.shape is ',points2.size())
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # dists's shape is [24, 64, 16]
            dists = square_distance(xyz1, xyz2)
            print('in the point propapation progress, dists.shape is ',dists.shape)
            dists, idx = dists.sort(dim=-1)
            # 
            print('after sort operation, dists.shape is ',dists.shape)
            print('after sort operation, idx.shape is ',idx.shape)
            print('dists is ',dists)
            print('idx is ',idx)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points


class GACNet(nn.Module):
    def __init__(self, num_classes,droupout=0,alpha=0.2):
        super(GACNet, self).__init__()
        # GraphAttentionConvLayer:        npoint, radius, nsample, in_channel, mlp,             group_all,  droupout,   alpha
        self.sa1 = GraphAttentionConvLayer(1024 ,   0.1 ,   32  ,   6 + 3   , [32, 32, 64]      , False,    droupout,   alpha)
        self.sa2 = GraphAttentionConvLayer(256  ,   0.2 ,   32  ,   64 + 3  , [64, 64, 128]     , False,    droupout,   alpha)
        self.sa3 = GraphAttentionConvLayer(64   ,   0.4 ,   32  ,   128 + 3 , [128, 128, 256]   , False,    droupout,   alpha)
        self.sa4 = GraphAttentionConvLayer(16   ,   0.8 ,   32  ,   256 + 3 , [256, 256, 512]   , False,    droupout,   alpha)
        # PointNetFeaturePropagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(droupout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    # pred = model(points[:,:3,:],points[:,3:,:])
    # Here, xyz is points[:,:3,:]
    # point is points[:,3:,:]
    def forward(self, xyz, point):
        # Here, self.sa1 is GraphAttentionConvLayer
        # points[:,:3,:],points[:,3:,:] is fed into GraphAttentionConvLayer function
        print('in GACNet, you will get xyz and point')
        print('xyz.shape and point.shape are')
        print('xyz.shape is',xyz.shape)
        print('point.shape is ',point.shape)
        # print('xyz is',xyz)
        # print('point is',point)        
        l1_xyz, l1_points = self.sa1(xyz, point)
        print('l1_xyz.shape is',l1_xyz.shape)
        print('l1_point.shape is ',l1_points.shape)
        print('l1_xyz is',l1_xyz)
        print('l1_point is',l1_points)        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        print('l2_xyz.shape is',l2_xyz.shape)
        print('l2_point.shape is ',l2_points.shape)
        print('l2_xyz is',l2_xyz)
        print('l2_point is',l2_points)           
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        print('l3_xyz.shape is',l3_xyz.shape)
        print('l3_point.shape is ',l3_points.shape)
        print('l3_xyz is',l3_xyz)
        print('l3_point is',l3_points)           
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        print('l4_xyz.shape is',l4_xyz.shape)
        print('l4_point.shape is ',l4_points.shape)
        print('l4_xyz is',l4_xyz)
        print('l4_point is',l4_points)   

        # this is upsampleing
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    model = GACNet(50)
    output = model(input)
function [node_exx,node_eyy,node_exy] = GetNodeStrain(meshInfo,exx,eyy,exy)
%%%===========================Copyright==================================%%%
	%%%   Version Nov. 2025
	%%%
	%%%   Gengxuan Zhu <zhugx@zju.edu.cn>
	%%%   Institute of Applied Mechanics,Zhejiang University
	%%%
	%%%===========================Description================================%%%
	%%% This is a function to calculate strain components of all nodes
	%%%======================================================================%%%    
    eleNodsID = meshInfo.eleNodsID;
    allNodeIDs = unique(eleNodsID(:));
    numNodes = length(allNodeIDs);
    nodesx = meshInfo.nodesx;
    nodesy = meshInfo.nodesy;
    
    node_exx = zeros(numNodes, 1);
    node_eyy = zeros(numNodes, 1);
    node_exy = zeros(numNodes, 1);
    % 遍历每个节点
    for i = 1:numNodes
        nodeID = allNodeIDs(i);
        [eleIdx, nodePos] = find(eleNodsID == nodeID);
        
        exx_vals = exx(sub2ind(size(exx), eleIdx, nodePos));  
        eyy_vals = eyy(sub2ind(size(eyy), eleIdx, nodePos));
        exy_vals = exy(sub2ind(size(exy), eleIdx, nodePos));
        
        % 计算加权系数（按单元面积加权，若没有面积则用简单平均）
        if exist('elemArea', 'var') && ~isempty(elemArea)
            weights = elemArea(eleIdx) / sum(elemArea(eleIdx));
        else
            weights = ones(size(eleIdx)) / length(eleIdx);
        end
        node_exx(i) = sum(exx_vals .* weights);
        node_eyy(i) = sum(eyy_vals .* weights);
        node_exy(i) = sum(exy_vals .* weights);
    end
    node_exx = reshape(node_exx,nodesx,nodesy)';
    node_eyy = reshape(node_eyy,nodesx,nodesy)';
    node_exy = reshape(node_exy,nodesx,nodesy)';
end
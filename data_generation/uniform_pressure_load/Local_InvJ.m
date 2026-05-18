function [locJ,detJ] = Local_InvJ(meshInfo,dNds,dNdt)
	%%%===========================Description================================%%%
	%%% This is a function to calculate the inverse Jacobia matrix and det 
    %%% at strain points per element.(|J| .* inv(J))
	%%%======================================================================%%%
    
    xEle = num2cell(meshInfo.xEle,1);
    yEle = num2cell(meshInfo.yEle,1);
    % Calculate entities of Jacobian Matrix
    J11 = cellfun(@(dNdt,yEle) dNdt .* yEle,dNdt,yEle,'UniformOutput',false);
    J11 = J11{1} + J11{2} + J11{3} + J11{4};
    J12 = cellfun(@(dNds,yEle) -dNds .* yEle,dNds,yEle,'UniformOutput',false);
    J12 = J12{1} + J12{2} + J12{3} + J12{4};
    J21 = cellfun(@(dNdt,xEle) -dNdt .* xEle,dNdt,xEle,'UniformOutput',false);
    J21 = J21{1} + J21{2} + J21{3} + J21{4};
    J22 = cellfun(@(dNds,xEle) dNds .* xEle,dNds,xEle,'UniformOutput',false);
    J22 = J22{1} + J22{2} + J22{3} + J22{4};
    locJ = {J11,J12,J21,J22};
    % Calculate det
    J11 = num2cell(J11);
    J12 = num2cell(J12);
    J21 = num2cell(J21);
    J22 = num2cell(J22);
    detJ = cellfun(@(J11,J12,J21,J22) J22 .* J11 - J21 .* J12,J11,J12,J21,J22,'UniformOutput',false);
end
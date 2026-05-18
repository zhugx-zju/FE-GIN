function femInfo = FEM_Assemble(meshInfo,materialInfo,BCInfo)
	%%%===========================Description================================%%%
	%%% This is a  function to get some matrix value of the GFE model
	%%%======================================================================%%%
    [Bgauss,w,detJgauss,Dgauss,Efield] = getFGMBD(meshInfo,materialInfo);
	[keVec,Ke] = getFGMKe(Dgauss,Bgauss,detJgauss,w);
    femInfo.Efield = Efield;
    femInfo.keVec = keVec;
    femInfo.Ke = Ke;
    femInfo.Bgauss = Bgauss;
    femInfo.detJgauss = detJgauss;
    femInfo.BCInfo = BCInfo; 
    femInfo.meshInfo = meshInfo;
end
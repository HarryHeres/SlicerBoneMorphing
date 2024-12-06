Eva
- [ ] upsample meshes
	- [ ] fill holes
	- [ ] upsample
	- [ ] clean
- [ ] double check point numbers pre and post crop to make sure no issues 

Jan
- [ ] add scaling factor in GUI
	- in global_registration:
		- Arthur used: 
		-  `distance_threshold = voxel_size * 1.5 
		- voxel_size = size / 55 (size is bounding box)`
		-  so our default distance_threshold in GUI should be size*1.5/55 
	
	
	- so to replicate Arthur's settings
		- if we do 0.007 factor (0.7%) of bounding box size for distance_threshold for ICP
		- then 0.027 (2.7%) of bounding box size for distance_threshold for Ransac
		- can interconvert, I think it makes sense for user to put it goal for ICP since that is the final one, and calculate Ransac from that



*Idea for setting max deformation allowed*

- check after rigid alignment max distance between two corresponding points (on cropped models only)
	- could use DeCa for this since the models are already aligned
- then use that to inform lambda value for deformation 
	- ie model allowed to deform enough to match proximal surfaces but distal surface can only deform by this amount
_Note_ hopefully this would not cause anisotropic morphing on the distal end, which could cause unrealistic epicondyle shapes. Maybe we can also constrain the distal end to not deform much in morphology 


*Other notes from Meeting 29/11/24*
- we discussed that SSMs will not solve our problem (as Jan already stated in thesis defense)
	- gives variation in length but would not necessarily help predict where within that feasible space the individual falls
		- indeed, other SSM methods for distal humerus reconstruction have issues with getting length exact
	- we would be happy to get similar errors to them
	- but we can do so by limiting deformation as discussed above
		- true length could only really be predicted if some part of proximal morphology can be used as predictor of distal
- future idea find closest match in proximal morphology (after scaling and rigid alignment)
	- check if finding closest match from "libary" of models used to create mean model gives better results than using mean model

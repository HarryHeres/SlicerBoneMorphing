#### Summary

- Eva showed examples alignment from Arthur's SSM analysis
	- we discussed rotational offset likely due to proximal alignment not being as good as whole bone alignment
		- Dave Ackland said if he remembers correctly in Huang et al SSM model, they did the partial reconstruction after aligning
			- so makes sense their method is a bit better (max error 35mm for distal humerus (point V), I did rough measure in Blender and found max error about 45mm)
			- still similar errors
		- advantage our method: more realistic case doing alignment with partial mesh ownly
- discussed combining SSM with deformation of proximal region to match known morphology of target as closely as possible
	- to deal with pathological subjects
	- for exact idea how to do this, see notes on To Do
	
- Arthur showed own dataset mice skulls comparing approaches gd and bcpd
	- bcpd probabilistic
	- gd less probabilistic
	- bcpd seems to perform better on outliers
	- want to test bgpd on our humerus dataset
	- ![BCPD vs GD comparison](Pasted%20image%2020250429180002.png)
- discussed open3d versions
	- probably I had old version installed when running Slicer which caused RANSAC issues whereas jupyter notebook with morphing env did not have issues
		- Eva can do some more tests 
			- note Arthur SSM reconstructions all had good alignment but maybe because pre aligned for DeCa
- note DeCa alignment had a rotational offset in bone F_RH_161 - check this

![Meeting discussion results](Pasted%20image%2020250429190156.png)


### TO DO

#### Eva
- [ ] send Jan an example of SSM reconstructed mesh for testing subsequent deformation
- [ ] send upsampling algorithm to Jan
	- goal to add as initial step to make partial target match 1/3 of mean mesh size
		- should also upsample mean mesh in case low res? could add that in too
- [ ] add leave one out to SSM code
- [ ] test SSM using non pre aligned meshes
- [ ] add landmarks to bone modelsnew repository open source
	- [ ] for both pathological and healthy
	- [ ] then we can add them to the SSM 
		- [ ] landmarks used for DeCa which is then used for sampling points for SSM
- [ ] check out weird deca alignment bone 
#### Jan
- [ ] add upsampling once Eva sends code
- [ ] test adding deformation only to specified region of statistical shape reconstructed model
	- [ ] idea to take bounding box of target (the 1/3 proximal humerus), grow by X amount (in case reconstructed a bit bigger than target in humeral head region)

#### Arthur
- [ ] send bcpd SSM method to us
- [ ] send tiny3d to Jan
### TO DO and Meeting Summary January '25
#### Summary
- Arthur successfully implemented SSM pipeline 
- works for partial meshes, ignoring points that are far away (achieved with RANSAC Regressor)
- distal reconstruction in examples looks good
- Arthur noted slight offset proximal rotational alignment would cause more distance offset distally
- Arthur tested different number of landmarks - in future can also tests effects landmark locations
- tps transform faster with fewer landmarks
	- note when changing # landmarks need to update number of PCs

**question from Eva**> should holes be filled automatically or no? we don't want to add extra vertices in there since it might mess with algorithm right? maybe fill with faces *after* resampling?
#### TO DO
**Eva:** 
- [ ] test SSM pipeline 
- [ ] compare outputs to paper using same dataset and SSMs
- [ ] bring Max onto project to do further landmarking to increase dataset?
- [ ] update remeshing functions to work with newest Slicer, and add optional smoothing
- [ ] testing RANSAC alignment criteria? how "strict" can we go? develop ideas starting point and then increments to increase by until alignment can be achieved

**Jan:**
- [ ] if time, RANSAC alignment optimization would be great


### Eva summary Ransac distance threshold tests

- default 2.7%, repeat 2x to check variation between runs
- ![[Pasted image 20250110180214.png]]


- here relative to target (magenta)
- ![[Pasted image 20250110180302.png]]

- Ransac = 1 (long axis rotation worse than 2.7 but rest is better aligned, same with distal alignment bc of rotation about other axis)

![[Pasted image 20250110180350.png]]
green is 2.7, yellow is 1
![[Pasted image 20250111203551.png]]
earlier test from RANSAC = 1 (better rotation alignment)

![[Pasted image 20250110180620.png]]


Ransac 0.5 in blue - again worse long axis rotation than higher ransac so overall still probably too high ransac value:) zellow was ransac ^1=

![[Pasted image 20250111210107.png]]

![[Pasted image 20250111220818.png]]
### TO DO and Meeting Summary Dec '24
Eva
- [x] upsample meshes
	- [x] fill holes
	- [x] upsample
	- [x] clean
- [x] double check point numbers pre and post crop to make sure no issues 

Jan
- [x] add scaling factor in GUI
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



*Other notes from Meeting 29/11/24*
- we discussed that SSMs will not solve our problem (as Jan already stated in thesis defense)
	- gives variation in length but would not necessarily help predict where within that feasible space the individual falls
		- indeed, other SSM methods for distal humerus reconstruction have issues with getting length exact
	- we would be happy to get similar errors to them
	- but we can do so by limiting deformation as discussed above
		- true length could only really be predicted if some part of proximal morphology can be used as predictor of distal
- future idea find closest match in proximal morphology (after scaling and rigid alignment)
	- check if finding closest match from "libary" of models used to create mean model gives better results than using mean model
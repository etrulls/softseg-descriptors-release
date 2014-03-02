This release includes the segmentation-aware descriptors of our CVPR 2013
paper [1]. It showcases SIFT and SID [2] with GbD embeddings [3]. We also used
PbD embeddings [4] which can be extracted from the Berkeley release on contour
detection [5]. This code should work as-is on any modern linux or OS X (tested
on Ubuntu variants and Mountain Lion, respectively).

Please cite [1] if you use this code on your research.

INSTRUCTIONS
1. Run compile.m on the main folder.
2. Run demo_sid.m and demo_sift.m. For SIFT you will also need VLFEAT [6].
Install it and edit line 3 to your installation folder.

NOTES
- SSID: It should take about a couple minutes to compute the descriptors and
about the same to compute the flow estimates. SID runs the DAISY framework
with as many convolution layers as scales, which requires a considerable
amount of memory. Use smaller images or sample more sparsely if you run into
problems. We intend to address this, so keep an out for further releases.
- SDSIFT: The results may differ from those in the paper as we reimplemented
it to increase performance. You may want to play with the size of the patch
used to determine the segmentation values at the center of the descriptor
(we average over the size of a SIFT bin).

REFERENCES
[1] E. Trulls, I. Kokkinos, A. Sanfeliu, F. Moreno-Noguer. Dense
segmentation-aware descriptors. CVPR 2013
[2] I. Kokkinos, A. Yuille. Scale invariance without scale selection. CVPR
2008.
[3] M. Leordeanu, R. Sukthankar, C. Sminchisescu. Efficient closed-form
solution to generalized boundary detection. ECCV 2012
[4] M. Maire, P. Arbelaez, C. Fowlkes, J. Malik. Using contours to detect
and localize junctions in natural images. CVPR 2008.
[5] http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_source.tgz
[6] http://www.vlfeat.org

CONTACT
Further updates at: http://www.iri.upc.edu/people/etrulls/#code
For details/questions please contact Eduard Trulls at etrulls@iri.upc.edu.

HISTORY
v0.2.1 (10 December 2013): Forgot one addpath call.
v0.2 (14 July 2013): Second release. Includes SDSIFT, and documents and
streamlines SSID.
v0.1 (4 July 2013): Initial release after CVPR. Includes SSID only.

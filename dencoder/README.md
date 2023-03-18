# Additional Details of dEncoder

## Illustration of Tree Pruning

dEncoder performs necessary merging and pruning to trim certain branches in the decision tree, which ultimately helps dataplane deployment. Here, we provide a visual illustration of this process in Figure 15. Based on the merging rules explained in Section 4.2 of the manuscript, the four leaf nodes at the bottom of Figure 15(a) can be merged into two nodes to produce a more concise tree in Figure 15(b). 

<div align=center>
    <img src=".\merge-figure2.png" height=500 />
</div>
<div align=center style="color:orange; 
    color: #999;
    padding: 2px;">Fig. 15. The process of merging and pruning decision trees.
</div>

## An Intuitive Explanation of Byte-Rules

Each decision tree branch is converted as one byte-rule. We provide an intuitive example to explain the design of byte-rule. Assume that a victim has experienced infiltration attacks (aiming to manipulate a compromised host to access the internal system) whereas an adversary launches Slowloris DoS that has not been seen by the victim. Infiltration traffic typically sets a relatively large TTL (e.g., 255) to ensure that more packets can penetrate the internal systems. Yet the benign traffic served by the victim (e.g., Web services) often sets a smaller bound on TTL to avoid loops (e.g., smaller than 64). 

The 22-th byte of a packet represents TTL. The goal of dLearner and dEncoder is to find a value range on the 22-th byte (denoted as $\mathcal B_{22}$) to differentiate benign and infiltration samples (recall the victim only experienced these two types of packets). There are many feasible ranges, such as [0:100] or [0:200]. However, since we prefer to profile the features of *benign traffic*, we will choose a range that is closer to benign traffic, e.g., $\mathcal B_{22} = [0, 64]$ . 

In Slowloris DoS attacks, the adversary tends to use a large TTL (e.g., 128) to keep the socket connection live. Thus, even if the victim has never seen Slowloris DoS (recall this is the defining trait of sophisticated attacks), the $\mathcal B_{22}$, trained on benign and infiltration samples, is still effective. Of course, this is a very simplified example to explain the effectiveness behind byte-rules. Thanks to the powerful fitting capabilities of the learning techniques, it is possible to obtain a set of byte-rules that can accurately describe the raw bytes of traffic desired by the victim. 

## Byte-Rule Selection

Since the number of matching rules installed in the DPDK dataplane is not unlimited. We discuss how to select these byte rules in case of limitation. In the straightforward case where the number of byte-rules in the benign rule list is lower than the matching rule limit, we can load all these benign rules into the dataplane. A similar case is that although the size of benign rule list is larger than the limit, the size of the attack rule list is below the limit. In this case, we can choose to load all attack byte-rules and label a low priority tag to those packets that match these rules. The two operations are equivalent since the two lists are obtained from *binary* decision tree. 

Finally, if the sizes of both lists are greater than the limit, we prefer to load the byte-rules that can match more benign traffic. To this end, we sort the benign rule list based on the tuple $\langle {{S_{num}},{H_{vol}}} \rangle$, where $S_{num}$ represents the number of benign samples covered by this rule in the decision tree, and $H_{vol}$ refers to the volume of the hyperplane occupied by this rule. The tuple for the byte-rule in Figure 15 is $S_{num}=2189$, and its ${H_{vol}} = {L_0} * {L_1} *  \cdots  * {L_i} = \left( {23 - 0 + 1} \right) * \left( {156 - 88 + 1} \right) * {256^{58}}$, where $L_i$ represents the number of integer values in the $i$-th byte range. The list is sorted in the descending order based on $S_{num}$ first and then $H_{vol}$. We select the first $n$ rules from the sorted list where $n$ is the limit. 

# Zooming into Preference Classifier

**(i) Malformed TCP packet attack.** From Figure 16(c), we can find that benign traffic forms several clusters, and the Malformed packets are distributed widely in the dimensionality-reduction space. This phenomenon is due to benign TCP packets universally conforming to the deterministic Mealy machine, such as using a three-way handshake to establish the connection. While the attacker will manually craft diverse flag combinations which be allowed. For example, the SYN-FIN flood constructs a series of Malformed packets which have SYN and FIN flags simultaneously. This SYN-FIN packet does not appear in the routine of benign operations because SYN is usually used to establish a connection while FIN is intended to terminate the connection. Therefore, the preference classifier could identify the unseen Malformed attack. 

**(ii) Fragment attack.** In Figure 16(d), it is clear that benign encrypted traffic presents two clusters while Fragment attack packets show a dispersed distribution. This is because that benign traffic usually doesn't set fragments or has a specific offset (e.g., related to MTU). Yet the attacker will set various offsets to cater to the attacking intent, such as reassembling the packets to enforce the port or IP address that is unallowed. Overall, the proposed preference learning is able to profile benign traffic without constantly chasing the emerging attacks. 

<div align=center>
    <img src=".\Jaqen.png" width=700 />
</div>
<div align=center style="color:orange; 
    color: #999; 
    padding: 2px;">Fig. 16. The DDoS defense policies in Jaqen.
</div>

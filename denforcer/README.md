# Additional Details of dEnforcer

We include additional details of dEnforcer. 

## Other Packet Scheduling Algorithms

Other schedulers, besides FastPath, can also be implemented in dEnforcer. For instance, an extended version of FastPath can append an early discard table after the fast forwarding table such that the packets that are not fast forwarded need to be filtered by the early discard table first. The early discard table maintains a list of source addresses with bad reputations, i.e., the vast majority of packets from these sources are classified as malicious. Packets matched by the early discard table are dropped immediately. This additional scheduler can save the downstream processing overhead at the ML-rule table and Q-pipeline otherwise consumed by these attack packets. Yet this scheduler may cause collateral damages to victim-desired clients sharing the same source addresses with adversaries. One design to compensate for this problem is only including sources that are not spoofable [LBKKKC19]. 

- [LBKKKC19] Luckie, M., Beverly, R., Koga, R., Keys, K., Kroll, J. A., & Claffy, K. (2019, November). Network hygiene, incentives, and regulation: deployment of source address validation in the Internet. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security (pp. 465-480).
- [PJZWAK14] Peter, S., Javed, U., Zhang, Q., Woos, D., Anderson, T., & Krishnamurthy, A. (2014). One tunnel is (often) enough. ACM SIGCOMM Computer Communication Review, 44(4), 99-110.
- [BRSPZHU15] Basescu, C., Reischuk, R. M., Szalachowski, P., Perrig, A., Zhang, Y., Hsiao, H. C., ... & Urakawa, J. (2015). SIBRA: Scalable internet bandwidth reservation architecture. arXiv preprint arXiv:1510.02696.

Other schedulers may extend the Q-pipeline with flexible queuing systems. For example, the victim can reserve a fraction of bandwidth for its premium clients using Weighted Fair Queuing such that these clients are less likely to be disrupted during DDoS attacks, achieving the similar purpose as described in [PJZWAK14, BRSPZHU15]. This, however, requires the M-pipeline to precisely identify the packets from premium clients. We leave the exploration of these schedulers to future work. 

<div align=center style="color:orange; 
    color: #999;
    padding: 2px;">Table 5. The parameter settings of dLearner and dEnforcer.
</div>

| Component |            Parameter            |                           Setting                            |                         Description                          |
| :-------: | :-----------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| dLearner  |             K-fold              |                           10-fold                            |                       Cross sampling.                        |
|           |              Loss               | $\lambda _{apl}:\lambda _{acc}:\lambda _{pre}:\lambda _{recall}:\lambda _{f1}=1:10:10:10:10$ |                        Loss function.                        |
|           |              Epoch              |                              80                              |                       Training epoch.                        |
| dEnforcer |             ML-rule             |                            1+60*1                            |                      ML-rule template.                       |
|           |      Fast forwarding table      |              benign>20,000 and benign/attack>10              | The conditions for adding the source address to the fast forwarding table. |
|           |        Validation cycle         |                        50,000 packets                        | The validation cycle (measured in packet number or time) of the fast forwarding table. |
|           |    Validating source address    |                              10                              | The number of source addresses checked in each validation cycle. |
|           | Empty the fast forwarding table |                            100ms                             | Empty the fast forwarding table when the high-priority queue in the Q-pipeline experiences consistent and severe congestion over a long period of time. |

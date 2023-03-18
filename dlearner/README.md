# Additional Details of dLearner

In this section, we include additional details of dLearner, including the anonymization of packet fields, loss function, and model training. Further, we suggest how to choose a decision tree or DNN for training dLearner. Finally, we also discuss how we address the common concerns about applying ML in network security [SP10, AQPFAFR20, A2000].

- [SP10] Sommer, R., & Paxson, V. (2010, May). Outside the closed world: On using machine learning for network intrusion detection. In *2010 IEEE symposium on security and privacy* (pp. 305-316). IEEE.
- [AQPFAFR20] Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., ... & Rieck, K. (2020). Dos and don'ts of machine learning in computer security. *arXiv preprint arXiv:2010.09470*.
- [A2000] Axelsson, S. (2000). The base-rate fallacy and the difficulty of intrusion detection. *ACM Transactions on Information and System Security (TISSEC)*, *3*(3), 186-205.

## Anonymize Data

In Section 4.1 of the manuscript, we discuss the process of data preprocessing before training the model. Particularly, we need to anonymize the specific bytes of the packets, such as source IP, MAC address, header checksum, etc. The model aims to learn the behavioral features of the packets rather than the address information, so anonymization is necessary to prevent overfitting. Specifically, the exact bytes that are zeroed for both TCP and UDP packets are shown in the below table. Note that our datasets only have IPv4 packets. 

<div align=center style="color:orange; 
    color: #999;
    padding: 2px;">Table 4. The bytes that require to be anonymized (zeroed) in TCP and UDP packets (with IPv4 at the IP-layer) to prevent overfitting during model training.
</div>

<div align=center>

Byte Positions | Representation   | Transport Protocol 
:-: | :-: | :-:  
0-11 | Src \& Dst MAC  | TCP \& UDP
18-19 | Identification  |  TCP \& UDP
24-25 | Header checksum  |  TCP \& UDP
26-33 | Src \& Dst IP  |  TCP \& UDP
34-35 |  Src port  | TCP \& UDP
40-41 | UDP checksum | UDP 
38-41 |  Sequence number |  TCP 
42-45 | Acknowledgment number | TCP
50-51 | TCP checksum | TCP
</div>

## Model Design

Based on training the surrogate model which is a method about the interpretability of neural networks [WHPSMVD18], we convert DNN into decision trees. To profile and learn desired traffic features, we guide model training with the following techniques. 

- [WHPSMVD18] Wu, M., Hughes, M., Parbhoo, S., Zazzi, M., Roth, V., & Doshi-Velez, F. (2018, April). Beyond sparsity: Tree regularization of deep models for interpretability. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).

<div align=center>
    <img src=".\DNN_to_Tree.png" height=500 />
</div>
<div align=center style="color:orange; 
    color: #999;
    padding: 2px;">Fig. 13. The overall of the model training process.
</div>

### Model Training

The overall training process is shown in the above figure. It contains three parts: training DNN, training Tree, and training surrogate model. 

**(i) Training DNN.** Use the true labels of the dataset to train a DNN model $M_{dnn}$, such as MLP or GRU. The loss function is ${loss}$ (explain subsequently), and the number of iterations is set to ${Epoch}_{1}$. 

**(ii) Training Tree.** After training the DNN, use the predicted labels of the DNN instead of the true labels of datasets to train a Gini decision tree $M_{tree}$. Based on Classification And Regression Tree (CART), we improve the tree algorithm that uses a benign-preferred CART to prefer learning benign traffic features. We introduce benign-preferred CART in the loss function section of the online repository (the loss function section). 

**(iii) Training Surrogate Model.** The surrogate model is essentially a Multilayer Perceptron (MLP). Input the weight matrix of DNN into surrogate model $M_{sur}$ to get the predicted metrics by $M_{sur}$. Note that the outputs by the surrogate model are $APL$, $ACC$, $F1$, $Precision$, $Recall$. Specifically, these metrics are expressed as follows. 

Assume that a combination $X = \{ {{x_1},{x_{2, \cdots ,}}{x_n}} \}$ with $n$ samples. The $APL$ can be calculated as follows:
$$APL = \frac{1}{n}\sum\limits_1^n {deep( {{x_i}} )}$$
where $deep\left( {x_i} \right)$ represents the path depth required by inputting ${x_i}$ into $M_{tree}$ to obtain the classification result. 

We define $ACC$ as the ratio between the number of correct predictions and the total number:
$$ACC = \frac{{{T{P_i} + T{N_i}} }}{{{{T{P_i} + T{N_i} + F{P_i} + F{N_i}}} }}$$
$Precision$ as the ratio of all positive samples in the correct prediction:
$$Precision = \frac{{{T{P_i}} }}{{{{T{P_i} + F{P_i}}} }}$$
and $Recall$ as the ratio of correctly predicted positive samples to all positive samples: 
$$Recall = \frac{{{T{P_i}} }}{{{{T{P_i} + F{N_i}}} }}$$
and the $F1$ score expression is:
$$F1 = \frac{{2 \cdot Precision \cdot Recall}}{{Precision + Recall}}$$
Where True Positive ($TP$) and False Positive ($FP$) indicate that positive samples are classified as positive and negative samples, respectively. True Negative ($TN$) and False Negative ($FN$) indicate that negative samples are classified as negative and positive samples, respectively. 

In this process, the loss function is the Mean Square Error (MSE) between the outputs predicted by $M_{sur}$ and the real metrics of the decision tree $M_{tree}$. And set an iteration number ${Epoch}_{2}$. 

The above three processes are collectively called a conversion from DNN to Tree. After converting ${Epoch}_{3}$ times, the learning of dLearner is completed. 

### Loss Function

We discuss the loss functions involved in the above training process in this part. 

**(i) DNN Loss Function.** In the training process of DNN model, we define the DNN loss function as 
$$Loss=DNN_H+\lambda_{apl} APL_{sur}-\lambda_{acc} ACC_{sur}-\lambda_{pre} Precision_{sur}-\lambda_{recall} Recall_{sur}-\lambda_{f1} F1_{sur}$$
where $DNN_H$ represents the crossentropy loss of DNN, $APL_{sur}$, $ACC_{sur}$, $Precision_{sur}$, $Recall_{sur}$, $F1_{sur}$ denote Average Path Length (APL), Accuracy, Precision, Recall and F1 score of the outputs by surrogate model, respectively. Moreover, coefficient $\lambda$ is the corresponding weight. 

These penalties in the loss function enable us to flexibly adapt the loss function based on our requirements. For instance, if the number of available matching rules in the dataplane is limited, we can properly increase ${\lambda_{apl}}$ to produce a relatively simpler decision tree. To support the FastPath scheduler design described in Section 4.3 of the menuscript, we need to increase ${\lambda_{recall}}$. 

**(ii) Benign-preferred CART.** Consider a decision tree. Denote the data at node $m$ as $Q_m$, with $N_m$ samples. According to the true labels, these samples can be divided into the attack sample set ${S_{attack}} = \{ {{x_{{a_1}}},{x_{{a_2}}}, \ldots, {x_{{a_t}}}} \}$ and the normal sample set ${S_{benign}} = \{ {x_{{b_1}}},{x_{{b_2}}}, \ldots ,{x_{b{({N_m} - t)}}} \}$, where $t$ represents the number of attack samples. We can get the average of the two sets:
$$AV{G_{attack}} = \frac{1}{t}\sum\limits_{u_{att} \in {S_{attack}}} u_{att}$$
$$AV{G_{benign}} = \frac{1}{{N_m} - t}\sum\limits_{u_{ben} \in {S_{benign}}} u_{ben}$$

For each candidate split $\theta = (j, t_m)$ consisting of a feature $j$ (i.e., a byte in packets) and threshold $t_m$, partition the data into $Q_m^{left}(\theta)$ and $Q_m^{right}(\theta)$ subsets
$$Q_m^{left}(\theta) = \{(x, y) | x_j <= t_m\}$$
$$Q_m^{right}(\theta) = Q_m \setminus Q_m^{left}(\theta)$$
The quality of a candidate split of node $m$ is then computed using an impurity function $H()$ 
$$G(Q_m, \theta) = \frac{N_m^{left}}{N_m} H(Q_m^{left}(\theta))+\frac{N_m^{right}}{N_m} H(Q_m^{right}(\theta)) \\+ {e^{ - \frac{{\left| t_m - AVG_{attack} \right|}}{{\left| t_m - AVG_{benign} \right|}}}}$$
Where ${e^{ - \frac{{\left| t_m - AVG_{attack} \right|}}{{\left| t_m - AVG_{benign} \right|}}}}$ is a penalty term that can make the split threshold $t_m$ closer to benign traffic. This supports the requirement that profiling benign traffic features and against sophisticated attacks. Particularly, we use the function $y = {e^{ - x}}$ to normalize the distance ratio. Specifically, it normalizes the range of $[0, \infty]$ to $(0, 1]$ (the function curve is shown in the below figure). 

Select the parameters that minimize the impurity
$$\theta^{\*}=\operatorname{argmin} G(Q_m, \theta)$$
Recurse for subsets $Q_m^{left}(\theta^{\*})$ and $Q_m^{right}(\theta^{\*})$ until the maximum allowable depth is reached or $N_m \le \min_{samples}$. 

In the experiment, we choose Gini as $H()$ and the calculation process is as follows. If a target is a classification outcome taking on values $0,1,\ldots,K-1$, for node $m$, let
$$p_{mk} = 1/ N_m \sum_{y \in Q_m} I(y = k)$$
be the proportion of class $k$ observations in node $m$. If $m$ is a terminal node, the predicted probability for this region is set to $p_{mk}$. Gini impurity is the following. 
$$H(Q_m) = \sum_k p_{mk} (1 - p_{mk})$$

**(iii) Surrogate Model Loss Function.**
The surrogate model loss is essentially an MSE, as follows 
$$Loss_{sur} = \frac{1}{5}[\left(APL_{tree}-APL_{sur}\right)^2+\left(ACC_{tree}-ACC_{sur}\right)^2+\left(F1_{tree}-F1_{sur}\right)^2+\left(Precision_{tree}-Precision_{sur}\right)^2+\left(Recall_{tree}-Recall_{sur}\right)^2]$$
where ${AP{L_{tree}}}$, ${AC{C_{tree}}}$, ${F{1_{tree}}}$, ${Precisio{n_{tree}}}$, ${Recal{l_{tree}}}$ represent Average Path Length, Accuracy, Precision, Recall and F1 score of the tree model, respectively. And ${AP{L_{sur}}}$, ${AC{C_{sur}}}$, ${F{1_{sur}}}$, ${Precisio{n_{sur}}}$, ${Recal{l_{sur}}}$ represent corresponding predicted outputs by the surrogate model, respectively. 

<div align=center>
    <img src=".\xx_font.png" height=500 />
</div>
<div align=center style="color:orange; 
    color: #999;
    padding: 2px;">Fig. 14. Function curve of $y = {e^{ - x}}$.
</div>

Particularly, the outputs of the surrogate model will be fed to the DNN training (calculate the loss) in the next conversion process (as shown in Figure 13). 

## Decision Tree or DNN?

We briefly discuss how to choose a decision tree or a DNN in dLearner. 

**Decision Tree.** Given a relatively simple pre-collected dataset that contains a small number of attack types, dLearner can directly train a decision tree model. Consider a site that provides ordinary services and supports a few types of applications. The decision tree may be capable of fitting and expressing the features of its data. In this case, we can directly train a decision tree since it has the advantage of a small training overhead. 

**DNN Model.** It is better to choose a DNN model instead of using a decision tree directly when the pre-collected dataset has complex distributions and contains many types of attacks. More importantly, the DNN model has better scalability. When against unknown sophisticated attacks or the victim has a very diverse user base, using DNN is conducive to learning potential patterns of benign traffic and performing better classification performance. In this case, it is unwise to directly train the decision tree model because its generalization ability is low. As for DNN, on the one hand, we can increase the depth of the model to improve the fitting ability; on the other hand, we can also use the sequential model (such as GRU) to fully understand the features of the traffic. As we demonstrated in Section 6.6, DFNet can accommodate a large number of ML rules with negligible forwarding overhead. Therefore, deploying complex DNN models in DFNet is promising. 

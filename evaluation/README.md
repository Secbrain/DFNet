# Additional Details of Evaluation

## Parameter settings

We list the main parameter settings of dLearner and dEnforcer in Table 5. 

## Dataset Division

Table 6 shows detailed packet samples of each group. 

<div align=center style="color:orange; 
    color: #999; 
    padding: 2px;">Table 6. The six groups of dataset division details for model-to-rule translation evaluation.
</div>
<div align=center>

| Divide | Set | Category | Sample |  | Concept drift | |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|     |      |    | Attack | Benign      | Attack | Benign |
| $d_1$      | Train        | UDP, SNMP, TFTP, Patator, MSSQL, WebDDoS, Heartbleed, Infiltration, UDP-Lag     | 1,862,660    | 2,727,656   | - | - |
|  | Test     | PortMap, NetBIOS, SSDP, SYN, LDAP, DNS, Botnet, Web-Attack, NTP, PortScan  | 1,423,574     | 2,479,217   | 465,665 | 681,913 |
| $d_2$   | Train   | Infiltration, Web-Attack, LDAP, Patator, UDP, MSSQL, Botnet | 1,026,437  | 2,930,170   | - | - |
|                         | Test                          | Heartbleed, NTP, SNMP, DNS, WebDDoS, PortMap, SSDP, SYN, TFTP, NetBIOS, UDP-Lag, PortScan | 2,468,853                            | 2,226,074                                   | 256,609 | 732,542 |
| $d_3$                   | Train                         | Patator, Heartbleed, PortScan, SNMP, WebDDoS, UDP, LDAP, NTP, DNS, TFTP, MSSQL, PortMap   | 1,799,631                            | 2,868,474                                   | - | - |
|                         | Test                          | NetBIOS, SYN, Infiltration, UDP-Lag, Botnet, SSDP, Web-Attack                             | 1,502,361                            | 2,303,194                                   | 449,907 | 717,118 |
| $d_4$                   | Train                         | DNS, Patator, NTP, UDP, Infiltration, LDAP, Botnet, PortScan, UDP-Lag, WebDDoS, TFTP      | 1,436,884                            | 2,211,236                                   | - | - |
|                         | Test                          | MSSQL, Heartbleed, PortMap, SNMP, NetBIOS, SSDP, Web-Attack, SYN                          | 1,955,795                            | 2,799,633                                   | 359,220 | 727,808 |
| $d_5$                   | Train                         | DNS, Heartbleed, NTP, PortScan, MSSQL, LDAP, UDP-Lag, SNMP, WebDDoS                       | 1,376,607                            | 2,191,854                                   | - | - |
|                         | Test                          | SYN, Botnet, Patator, Web-Attack, NetBIOS, PortMap, TFTP, UDP, SSDP, Infiltration         | 2,031,141                            | 2,979,478                                   | 344,151 | 547,963 |
| $d_6$                   | Train                         | SSDP, PortMap, DNS, NTP, SNMP, Web-Attack, TFTP, SYN, UDP-Lag                             | 1,226,753                            | 1,722,460                                   | - | - |
|                         | Test                          | UDP, Infiltration, Patator, Heartbleed, MSSQL, Botnet, PortScan, NetBIOS, LDAP, WebDDoS   | 2,218,458                            | 3,096,826     | 306,688 |  430,615  |
</div>

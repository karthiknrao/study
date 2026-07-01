# CMU 15-892 Foundations of Electronic Marketplaces (Fall 2015) — Reading List

Source: http://www.cs.cmu.edu/~sandholm/cs15-892F15/cs15-892.htm

All paper/document links listed on the course page **after** the lecture-schedule block. Organised by the page's section headings. Page-relative paths are resolved against the course directory (`http://www.cs.cmu.edu/~sandholm/cs15-892F15/`) and `../`-prefixed paths against `http://www.cs.cmu.edu/~sandholm/`.

## TOPICS

### General review articles

- [Computing in Mechanism Design . by T. Sandholm. In the New Palgrave Dictionary in Economics, 2008 . ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/computing%20in%20mech%20design.Palgrave.pdf)
- [Computational Mechanism Design by David C. Parkes. In Lecture notes of Tutorials at 10th Conf. on Theoretical Aspects of Rationality and Knowledge (TARK-05) , Institute of Mathematical Sciences, University of Singapore, 2008. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ParkesTARK08.pdf)
- ["Combinatorial Auctions (a survey)" by Blumrosen and Nisan. Chapter 11 of the book Algorithmic Game Theory .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/algorithmic-game-theory.pdf)
- [�Auctions: Theory� by Lawrence Ausubel. � To appear in the New Palgrave Dictionary of Economics . ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Ausubel_Auction_Theory_Palgrave.pdf)

### Basics of mechanism design

- Nisan, N. 2007. Introduction to Mechanism Design (for Computer Scientists) . Chapter 9 of the book Algorithmic Game Theory .
- Mas-Colell, Whinston & Green. Microeconomic theory. , Chapter 23. Oxford University Press, 1995. (Includes the Myerson-Satterthwaite theorem, but does not cover virtual implementation).
- [Review article [Parkes 01 (PS)] [bibliography for this article (PS)]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Parkes_mechanism_design_review.ps)
- [Review article Implementation Theory (PDF) by Maskin & Sjostrom, 2001 (Does not cover dominant strategy implementation; first 80% is for complete information environments; focuses on implementation that does not have bad equilibria also).](http://www.cs.cmu.edu/~sandholm/cs15-892F15/implementation%20theory%20-no%20dom%20strat.pdf)
- Osborne and Rubinstein. A Course in Game Theory, MIT Press, 1994.

### In designing (exact or approximate) mechanisms, it can help to know what mechanism families are incentive compatible, and what is (im)possible:

- [Truthful germs are contagious: A local to global characterization of truthfulness. A. Archer and R. Kleinberg, EC-08.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/truthful%20germs.EC08.pdf)
- [A Modular Approach to Roberts' Theorem . Dobzinski and Nisan, SAGT-09.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ModularSAGT09.pdf)
- [Two Simplified Proofs for Roberts' Theorem . Ron Lavi, Ahuva Mu'alem, and Noam Nisan. Social Choice and Welfare , 32, pp. 407 -- 423, 2009.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/two-simplified-proofs-Lavi09.pdf)
- [The Limits of Ex Post Implementation. Philippe Jehiel, Moritz Meyer-ter-Vehn, Benny Moldovanu, and William R. Zame Econometrica , 2006.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/limits%20of%20ex%20post%20implementation.pdf)
- [Ex post implementation. Dirk Bergemann and Stephen Morris. Games and Economic Behavior 63 (2008) 527-566.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ex%20post%20implementation.GEB08.pdf)
- [Multi-Unit Auctions with Budget Limits . Shahar Dobzinski, Ron Lavi, and Noam Nisan. FOCS-08.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/multi-unitFOCS08.pdf)
- [On Characterizations of Truthful Mechanisms for Combinatorial Auctions and Scheduling . Shahar Dobzinski and Mukund Sundararajan. EC-08. See also the addendum .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Truthful-MechanismsEC08.pdf)
- [Paths, Cycles and Mechanism Design , by Vohra, 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/paths,%20cycles%20and%20MD.pdf)
- [Weak Monotonicity characterizes deterministic dominant strategy implementation by S. Bikhchandani, S. Chatterji, R. Lavi, A. Mu'alem, N. Nisan, and A. Sen. Econometrica , 74(4), pp. 1109 -- 1132, 2006. See also the supplementary material for this paper.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/weak-monotonicity-econometrica06.pdf)
- [Characterization of Revenue Equivalence , by B. Heydenreich, Rudolf Muller, Marc Uetz, and Rakesh Vohra, 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/revenue_equivalence_Heydenreich07.pdf)
- [Characterizing Dominant Strategy Mechanisms with Multi-Dimensional Types . [Gui, Mueller, Vohra 2004 draft]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Dominant-Strategy-Gui04.pdf)
- [Truthful Mechanism Design for Multi-Dimensional Scheduling via Cycle Monotonicity . Ron Lavi and Chaitanya Swamy. EC-07.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Truthful-Mechanism-Design-EC07.pdf)

### Auctioning a single item

- [Mechanism Design and Approximation. Draft of book by Jason Hartline. Available for free. Especially Chapter 3.](http://jasonhartline.com/MDnA/)
- Auction Theory. Book by Vijay Krishna. (I don't think this is available for free, but it is available on Amazon.)
- [Profit Maximization in Mechanism Design. By Hartline and Karlin. Chapter 13 of the book Algorithmic Game Theory . Sections 13.1-13.2.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/algorithmic-game-theory.pdf)
- [Bayesian Optimal No-deficit Mechanism Design . By Shuchi Chawla, Jason Hartline, Uday Rajan and R. Ravi, WINE'06.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Bayesian-Optimal-No-Deficit-WINE06.pdf)
- [Review article Auctions: An Introduction , Wolfstetter 1994](http://www.cs.cmu.edu/~sandholm/cs15-892F15/auctionsurvey.ps)
- Advanced material on non-private value auctions [Dasgupta & Maskin QJE-00], [Jehiel & Moldovanu 1998]

### Optimal (batch) clearing of multi-item and/or multi-unit markets

- Sandholm, T. 2006. Optimal Winner Determination Algorithms. Chapter 14 of the book Combinatorial Auctions , Cramton, Shoham, and Steinberg, eds., MIT Press.
- [Sandholm, T. 2013. Very-Large-Scale Generalized Combinatorial Multi-Attribute Auctions: Lessons from Conducting $60 Billion of Sourcing. Chapter 16 in The Handbook of Market Design , edited by Nir Vulkan, Alvin E. Roth, and Zvika Neeman, Oxford University Press.](http://www.cs.cmu.edu/~sandholm/Expressive%20commerce.Market%20Design%20book.v2.pdf)
- [Sandholm, T., Suri, S., Gilpin, A., and Levine, D. 2005. CABOB: A Fast Optimal Algorithm for Winner Determination in Combinatorial Auctions. Management Science 51(3), 374-390.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/CABOB.MgmtSci05.pdf)
- [Lehmann, D., Mueller, R., and Sandholm, T. 2006. The Winner Determination Problem. Chapter 12 of the book Combinatorial Auctions , Cramton, Shoham, and Steinberg, eds., MIT Press. A Kernel Method for Market Clearing. By Sebastien Lahaie. IJCAI-09.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/winner-determination-final.pdf)
- [Gilpin, A. and Sandholm, T. 2011. Information-Theoretic Approaches to Branching in Search. Discrete Optimization , 8, 147-159.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/entropic.DiscreteOptimization.pdf)
- [Bidding and allocation in combinatorial auctions [Nisan EC-00]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Bidding%20and%20allocation%20in%20CAs.ps)
- Winner determination in combinatorial auction generalizations. [Sandholm et al AAMAS-02]
- Side constraints and non-price attributes in markets. [Sandholm et al IJCAI-01 workshop: Distributed constraint reasoning] ; short later version appeared in Games and Economic Behavior .
- [Computational complexity of clearing exchanges with supply-demand curves [Sandholm-Suri ISAAC-01]](http://www.cs.cmu.edu/~sandholm/supply-demand.aaai02WS.pdf)
- [Computational complexity of clearing multi-unit auctions [Sandholm-Suri IJCAI-01]](http://www.cs.cmu.edu/~sandholm/clearability.ijcai01.pdf)
- [Fast Vickrey-Clarke-Groves computation in networks [Suri-Hirschberg FOCS-01]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Hershberger.ps)

### Expressiveness of mechanisms

- A Theory of Expressiveness in Mechanisms. Michael Benisch and Tuomas Sandholm. Draft, 2011. Very short early version in appeared in AAAI-08. Very-Large-Scale Generalized Combinatorial Multi-Attribute Auctions: Lessons from Conducting $60 Billion of Sourcing. Sandholm, T. Chapter 16 in The Handbook of Market Design , edited by Nir Vulkan, Alvin E. Roth, and Zvika Neeman, Oxford University Press.
- [P. Dütting, F. Fischer, D. C. Parkes, Expressiveness and Robustness of First-Price Position Auctions , EC’14](http://paulduetting.com/pubs/ec14-gfp.pdf)
- [Simplicity-Expressiveness Tradeoffs in Mechanism Design. Paul Dütting, Felix Fischer, and David Parkes. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/simplicity-expressiveness tradeoffs in MD.ec11.pdf)
- Milgrom, P. Simplified Mechanisms with an Application to Sponsored-Search Auctions. Games and Economic Behavio r, Sept 2010, vol 70, Issue 1: 62-70..
- [Multi-Keyword Sponsored Search. Peerapong Dhangwatnotai. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/multi-keyword sponsored search.ec11.pdf)
- [Benisch, M., Sadeh, N., and Sandholm, T. Methodology for Designing Reasonably Expressive Mechanisms with Application to Ad Auctions In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) . ( PDF )](http://www.cs.cmu.edu/~sandholm/methodologyReasonablyExpressive.ijcai09.pdf)
- Sandholm, T. 2007. Expressive Commerce and Its Application to Sourcing: How We Conducted $35 Billion of Generalized Combinatorial Auctions. AI Magazine, 28(3), 45-58.
- [Boutilier, C., Parkes, D., Sandholm, T., and Walsh, W. 2008. Expressive Banner Ad Auctions and Model-Based Online Optimization for Clearing. In Proceedings of the National Conference on Artificial Intelligence (AAAI) . ( PDF )](http://www.cs.cmu.edu/~sandholm/expressiveBannerAdAuctions.AAAI08.pdf)
- [Position Auctions with Budgets: Existence and Uniqueness . By Itai Ashlagi, Mark Braverman, Avinatan Hassidim, Ron Lavi, and Moshe Tennenholtz.Draft 2009. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/position-auctions-with-budgets09.pdf)
- [Optimize-and-Dispatch Architecture for Expressive Ad Auctions by David C. Parkes and Tuomas Sandholm. In the Proceedings of First Workshop on Sponsored Search Auctions, 2005.( PDF )](http://www.cs.cmu.edu/~sandholm/optimize-and-dispatch.ws05.pdf)
- [On Expressing Value Externalities in Position Auctions . Florin Constantin, Malvika Rao, Chien-Chung Huang, and David C. Parkes. AAAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ExpressingExternalitiesInPositionAuctions.aaai11.pdf)
- [Externalities in Keyword Auctions: An Empirical and Theoretical Assessment , R. Gomes, N. Immorlica and E. Markakis, in Proc. 5th Workshop on Ad Auctions (2009). ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/externalities09.pdf)
- [Sponsored Search with Contexts. Eyal Even-Dar, Michael Kearns, and Jennifer Wortman. WWW-07.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/sponsored search with contexts.www07.pdf)

### Multi-stage market designs with preference elicitation

- Preference elicitation in combinatorial auctions [Sandholm-Boutilier Chapter 10 in the book �Combinatorial Auctions�, 2006]
- [Iterative combinatorial auctions (iBundle etc.) [Parkes�s chapter in the book �Combinatorial Auctions� 2006] [OLD: Parkes ACM-EC-99 , AAAI-00a , AAAI-00b ]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/iBundle.ps)
- [A Kernel-Based Iterative Combinatorial Auction . Sebastien Lahaie, EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Kernel-basedIterativeCA.AAAI11.pdf)
- The Communication Requirements of Combinatorial Allocation Problems. By Ilya Segal, Chapter 11 of the book Combinatorial Auctions , 2006.
- [Ascending Price Vickrey Auctions for General Valuations ( PDF ) by Debasis Mishra and David C. Parkes. Journal of Economic Theory 132, 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Ascending-Price-Vickrey.pdf)
- [Exponential Communication Inefficiency of Demand Queries by N. Nisan and I. Segal. TARK 2005 .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Exponential-Communication-TARK05.pdf)
- [Multi-Item Vickrey-Dutch Auctions ( PDF ) Draft by Debasis Mishra and David C. Parkes, 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Multi-Item-Vickrey-Dutch.pdf)
- [Communication complexity of approximate set packing and covering [Nisan 01]](http://cs.cmu.edu/~sandholm/communication%20in%20set%20cover%20and%20packing.ps)
- Linear programming and Vickrey auc tions [Vohra et al. draft 01]
- [Dynamic auction for multiple distinguishable items [Ausubel 00] (slides from Nisan�s course)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Ausubel%20multi-item%20auction.pdf)
- [AkBA [Wurman et al ACM-EC-00]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/AkBA.ps)
- [Auction Design with Costly Preference Elicitation ( PDF ) by David C. Parkes. In Annals of Mathematics and AI 44, 2005, pages 269-302.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/costlyprefs.pdf)

### Automated mechanism design (AMD)

#### ...for the general problem

- Conitzer, V. and Sandholm, T. 2002. Complexity of Mechanism Design. In Proceedings of the 18th Conference on Uncertainty in Artificial Intelligence (UAI) .
- [Sandholm, T. 2003. Automated mechanism design: A New Application Area for Search Algorithms. In Proceedings of the International Conference on Principles and Practice of Constraint Programming (CP) .](http://www.cs.cmu.edu/~sandholm/amd_overview.cp03.pdf)
- [Conitzer, V. and Sandholm, T. 2004. Self-Interested Automated Mechanism Design and Implications for Optimal Combinatorial Auctions. In Proceedings of the ACM Conference on Electronic Commerce (EC), pp. 132-141 . Sui, X., Boutilier, C., and Sandholm, T. 2013. Analysis and Optimization of Multi-dimensional Percentile Mechanisms. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) .](http://www.cs.cmu.edu/~sandholm/SI_AMD_and_CAs.acmec04.pdf)
- [Sandholm, T., Conitzer, V., and Boutilier, C. 2007. Automated Design of Multistage Mechanisms. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) .](http://www.cs.cmu.edu/~sandholm/multistageAMD.ijcai07.pdf)
- [Conitzer, V. and Sandholm, T. 2007. Incremental Mechanism Design. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) .](http://www.cs.cmu.edu/~sandholm/incrementalAMD.ijcai07.pdf)
- [Conitzer, V. and Sandholm, T. 2004. An Algorithm for Automatically Designing Deterministic Mechanisms without Payments. In Proceedings of the International Joint Conference on Autonomous Agents and Multiagent Systems (AAMAS) , pp. 128-135, New York, July 19-23.](http://www.cs.cmu.edu/~sandholm/amd_algorithm.AAMAS04.pdf)
- [Conitzer, V. and Sandholm, T. 2003. Applications of automated mechanism design. In Proceedings of the UAI Bayesian Modeling Applications Workshop , Acapulco, Mexico. Extended version.](http://www.cs.cmu.edu/~sandholm/applications_of_AMD.uaiWS03.pdf)
- [Conitzer, V. and Sandholm, T. 2003. Automated mechanism design with a structured outcome space. Draft.](http://www.cs.cmu.edu/~sandholm/AMD_structured.draft.pdf)
- [Conitzer, V. and Sandholm, T. 2003. Automated Mechanism Design: Complexity Results Stemming from the Single-Agent Setting. In Proceedings of the International Conference on Electronic Commerce (ICEC) , Pittsburgh, September 30 � October 3.](http://www.cs.cmu.edu/~sandholm/1-agent%20AMD.icec03.pdf)

#### ...for auctions and other selling mechanisms

- [Sandholm, T. and Likhodedov, A. 2015. Automated Design of Revenue-Maximizing Combinatorial Auctions. Operations Research 63(5), 1000-1025. (Subsumes and extends over a AAAI-05 paper and a AAAI-04 paper .)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/approximating.aaai05.pdf)
- [Daskalakis, C. Multi-Item Auctions Defying Intuition? Newsletter of the ACM Special Interest Group on E-commerce, 14(1), 2015. pdf](http://www.sigecom.org/exchanges/volume_14/1/)
- [Cai, Y., Daskalakis, C., and Weinberg, M. Reducing Bayesian Mechanism Design to Algorithm Design. Encyclopedia of Algorithms, 2015. pdf](http://link.springer.com/referenceworkentry/10.1007/978-3-642-27848-8_787-1)
- [Sandholm, T. and Gilpin, A. 2003. Sequences of Take-It-or-Leave-It Offers: Near-Optimal Auctions without Full Valuation Revelation. In Proceedings of the AAMAS workshop on Agent-Mediated Electronic Commerce (AMEC V) , Melbourne,Australia.](http://www.cs.cmu.edu/~sandholm/take_it.amec03.pdf)
- [Cai, Y., Daskalakis, C., and Weinberg, M. 2013. Understanding Incentives: Mechanism Design becomes Algorithm Design. In FOCS .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/understanding incentives.focs-13.pdf)
- [Cai, Y., Daskalakis, C., and Weinberg, M. 2012. Optimal Multi-Dimensional Mechanism Design: Reducing Revenue to Welfare Maximization. In FOCS . arxiv](http://arxiv.org/abs/1207.5518)
- [Tang, P. and Sandholm, T. 2012. Mixed-bundling auctions with reserve prices. In Proceedings of the International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS) .](http://www.cs.cmu.edu/~sandholm/MixedBundlingWReservePrices.aamas12.pdf)
- [Tang, P. and Sandholm, T. 2011. Approximating Optimal Combinatorial Auctions for Complements Using Restricted Welfare Maximization. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) .](http://www.cs.cmu.edu/~sandholm/approximatingLevinBasedOnWelfareMaximization.IJCAI11.pdf)
- [Othman, A. and Sandholm, T. 2009. How Pervasive is the Myerson-Satterthwaite Impossibility? In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) .](http://www.cs.cmu.edu/~sandholm/myersat.ijcai09.pdf)
- [On approximating optimal auctions [Ronen EC-01]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/www.cs.cmu.edu/~sandholm/cs15-892F07/aproxAuct6.ps)

#### ...for other applications

- [R. Jurca and B. Faltings. Collusion Resistant, Incentive Compatible Feedback Payments . Proceedings of the ACM Conference on E-Commerce (EC'07) , pp. 200-209, San Diego June 11-15 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Collusion-resistantEC2007.pdf)
- [R. Jurca and B. Faltings. Minimum Payments that Reward Honest Reputation Feedback . Proceedings of the ACM Conference on Electronic Commerce (EC2006) , pp. 190-199, Ann Arbor, Michigan, June 11-15 2006.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/MinimumPaymentsReward_EC2006.pdf)

### Auction and exchange design without priors

- Mechanism Design via Machine Learning. [Balcan, Blum, Hartline, Mansour], JCSS, 2008.
- [Competitive generalized auctions [Fiat, Goldberg, Hartline, Karlin]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/auctions-STOC-02.pdf)
- [Truthful and Competitive Double Auctions [Deshmukh, Goldberg, Hartline, Karlin]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/double-auctions-ESA-02.pdf)
- Pricing without demand curves [Segal, American Economic Review ]
- Market research and Market Design [Vohra & Baliga]
- [[OLD READINGS ( paper1 , paper2 ) (slides from Nisan�s course) ]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Competitive%20auctions%20and%20digital%20goods.ps)

### Incentive-compatible (IC) approximation by the auctioneer

- [Multi-Unit Auctions: Beyond Roberts. Shahar Dobzinski and Noam Nisan. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/multi-unit auctions - beyond Roberts.ec11.pdf)
- [VC v. VCG: Inapproximability of Combinatorial Auctions via Generalizations of the VC Dimension. Elchanan Mossel, Christos Papadimitriou, Michael Schapira, and Yaron Singer.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/VC v VCG.pdf)
- [Computation and Incentives in Combinatorial Public Projects. Dave Buchfuhrer, Michael Schapira, and Yaron Singer.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Computation and incentives in combinatorial public projects.pdf)
- [Bayesian Mechanism Design for Budget-Constrained Agents. Shuchi Chawla, David Malec, and Azarakhsh Malekian. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Bayesian MD for budget-constrained agents.ec11.pdf)
- Computationally-Efficient Approximation Mechanisms . Chapter by Lavi in the book Algorithmic Game Theory .
- [Algorithmic mechanism design [Nisan-Ronen GEB 2001]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/selfishJ.pdf)
- [Truth revelation in rapid approximately efficient combinatorial auctions [Lehman-O�Callaghan-Shoham JACM-02]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Lehmann%20et%20al%20JACM-02.pdf)
- [Truthful and Near-optimal Mechanism Design via Linear Programming , by Ron Lavi and Chaitanya Swamy (early version in FOCS-05).](http://www.cs.cmu.edu/~sandholm/cs15-892F15/mechdeslp-journ-1.pdf)
- [On the Power of Randomization in Algorithmic Mechanism Design , FOCS-09. Shahar Dobzinski and Shaddin Dughmi.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/randompower-current.pdf)
- [Truthful Randomized Mechanisms for Combinatorial Auctions by S. Dobzinski, N. Nisan, and M. Schapira. STOC 2006 .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/randomca-1.pdf)
- [Impersonation-Based Mechanisms , By Moshe Babaioff, Ron Lavi, and Elan Pavlov, AAAI-06.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/impersonationAAAI06.pdf)
- [Two Randomized Mechanisms for Combinatorial Auctions by S. Dobzinski, APPROX-08.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/TruthfulRandomizedCombinatorialAuctionsSTOC06.pdf)
- [Limitations of VCG-based Mechanisms by S. Dobzinski and N. Nisan. STOC 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/limit-vcg-STOC07.pdf)
- [Mechanisms for Multi-Unit Auctions by S. Dobzinski and N. Nisan. EC-07.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/MechanismsMulti-UnitAuctionsEC07.pdf)
- [Algorithms for selfish agents [Nisan 01]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Algorithms%20for%20selfish%20agents%20-%20Nisan.ps)
- [Computationally feasible VCG mechanism [Nisan-Ronen 00]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Computationally%20feasible%20VCG%20mechanisms.ps)
- [Algorithms for rational agents [Ronen] � Section 7 (if not subsumed by Ronen�s EC-01 paper)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Algs%20for%20rational%20agents%20-%20use%20sec%207.ps)
- [Mechanism design for resource-bounded agents [Monderer-Tennenholtz-Kfir Dahav ICMAS-00]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Monderer%20ICMAS-00.ps)

### Bidding agents with hard valuation problems

- [Dominant-Strategy Auction Design for Agents with Uncertain, Private Values. David R. M. Thompson and Kevin Leyton-Brown. AAAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/DSE Auction for uncertain private values.aaai-11.pdf)
- Efficient Metadeliberation Auctions. Cavallo and Parkes, AAAI-08.
- Larson, K. and Sandholm, T. 2005. Mechanism Design and Deliberative Agents. Proceedings of the International Joint Conference on Autonomous Agents and Multi-Agent Systems (AAMAS)
- Larson, K. and Sandholm, T. 2001. Costly Valuation Computation in Auctions. In Proceedings of the Theoretical Aspects of Reasoning about Knowledge (TARK)
- [Larson, K. and Sandholm, T. 2001. Computationally Limited Agents in Auctions. In Proceedings of the International Conference on Autonomous Agents, Workshop on Agent-based Approaches to B2B .](http://www.cs.cmu.edu/~sandholm/computationally_limited.agents01ws.pdf)
- [Issues in computational Vickrey auctions [Sandholm IJEC-00 (originally ICMAS-96)]](http://www.cs.cmu.edu/~sandholm/vickrey.IJEC.ps)
- Valuation complexity explains last-minute bidding [Eric Rasmusen draft-03]
- [Computationally feasible VCG mechanism [Nisan-Ronen 00] (This paper contains the second-chance mechanism and the maximal-in-range approach.)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Computationally%20feasible%20VCG%20mechanisms.ps)
- [Ben-Sasson, E., Kalai, A., and Kalai E. An Approach to Bounded Rationality. NIPS . (This is not really about valuation calculation, but has some results about strategies with costs.)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/an_approach_to_bounded_rationality.pdf)

### Avoiding manipulation using computational complexity; Mechanism design for computationally limited agents; Non-truth-promoting mechanisms

- [Approximately Strategy-Proof Voting. Eleanor Birrell and Rafael Pass. IJCAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/approximately strategy-proof voting.ijcai11.pdf)
- Othman, A. and Sandholm, T. 2009. Better with Byzantine: Manipulation-Optimal Mechanisms. In Proceedings of the Symposium on Algorithmic Game Theory (SAGT) .
- [Conitzer, V. and Sandholm, T. 2003. Computational Criticisms of the Revelation Principle. In Proceedings of the Workshop on Agent Mediated Electronic Commerce (AMEC V) . Newer draft.](http://www.cs.cmu.edu/~sandholm/revelation.draft_around_LOFT04_time.pdf)
- Conitzer, V., Sandholm, T., and Lang, J. 2007. When Are Elections with Few Candidates Hard to Manipulate? Journal of the ACM , 54(3).
- Conitzer, V. and Sandholm, T. 2003. Universal Voting Protocol Tweaks to Make Manipulation Hard. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI).
- Conitzer, V. and Sandholm, T. 2006. Nonexistence of Voting Rules That Are Usually Hard to Manipulate. In Proceedings of the National Conference on Artificial Intelligence (AAAI)
- [Ariel D. Procaccia and Jeffrey S. Rosenschein. 2007. Junta Distributions and the Average-Case Complexity of Manipulating Elections. Journal of Artificial Intelligence Research. Volume 28, pages 157-181. [ PDF ]](http://www.cs.cmu.edu/~sandholm/cs15-892F15/JuntaDistributionsJAIR07.pdf)
- [The Geometry of Manipulation - a Quantitative Proof of the Gibbard Satterthwaite Theorem. Marcus Isaksson, Guy Kindler, and Elchanan Mossel. FOCS-10. This is for 4 or more candidates.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Geometry of manipulation.2010.pdf)
- [A Quantitative Version of the Gibbard-Satterthwaite theorem for Three Alternatives . E. Friedgut, G. Kalai, N. Keller and N. Nisan. Preliminary version titled "Elections can be Manipulated Often" appeared in FOCS 2008.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/mani2.pdf)

### Online mechanisms

- Online Mechanisms, by David Parkes. Chapter in Algorithmic Game Theory .
- Self-Correcting Sampling-Based Dynamic Multi-Unit Auctions. By Florin Constantin and David C. Parkes. Bonn workshop on mechanism design, 2009.
- [Self-Correcting Sampling-Based Dynamic Multi-Unit Auctions . by Florin Constantin and David C. Parkes. EC-09.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Self-CorrectingEC09.pdf)
- [Learning About The Future and Dynamic Efficiency , Alex Gershkov and Benny Moldovanu, in American Economic Review .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/learning about future and dynamic efficiency.pdf)
- [Dynamic Revenue Maximization with Heterogeneous Objects: A Mechanism Design Approach. Alex Gershkov and Benny Moldovanu, American Economic Journal: Microeconomics, Vol. 1, No. 2, 2009.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/dynamic revenue maximization with heterogeneous objects.pdf)
- [An Efficient Dynamic Mechanism , Susan Athey and Ilya Segal, 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/EfficientDynamic.Athey and Segal 07.pdf)
- [The Dynamic Pivot Mechanism . Dirk Bergemann and Juuso Välimäki. Econometrica, (2010) 78: 771-789.](http://dirkbergemann.commons.yale.edu/files/2011/01/Paper29_ECTA7260-dynamicpivot.pdf)
- [Efficient Sequential Assignment with Incomplete Information , Alex Gershkov and Benny Moldovanu, in Games and Economic Behaviour.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/EfficientSequential09.pdf)
- [Efficiency Levels in Sequential Auctions with Dynamic Arrivals Lavi and Segev. Draft 2009.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/sequential-auctions24.pdf)
- [Prompt Mechanisms for Online Auctions . Richard Cole, Shahar Dobzinski, and Lisa Fleischer. SAGT-08](http://www.cs.cmu.edu/~sandholm/cs15-892F15/PromptMechanismsSAGT08.pdf)
- [Online algorithms for market clearing Blum-Sandholm-Zinkevich. JACM, 2006](http://www.cs.cmu.edu/~sandholm/online_clearing.jacm.pdf)
- [Automated Online Mechanism Design and Prophet Inequalities. Hajiaghayi, M., Kleinberg, R., and Sandholm, T., AAAI-07 .](http://www.cs.cmu.edu/~sandholm/www/prophet.aaai07.pdf)
- [An Ironing-Based Approach to Adaptive Online Mechanism Design in Single-Valued Domains by David C. Parkes and Quang Duong, AAAI-07.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Ironing-BasedApproachaaai07.pdf)
- [Chain: A dynamic double auction framework by Jonathan Bredin, David C. Parkes, and Quang Duong. In Journal of Artificial Intelligence Research, 2007.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Chain-DynamicDouble07.pdf)
- [Competitive Analysis of Incentive Compatible On-Line Auctions by Ron Lavi and Noam Nisan. Theoretical Computer Science 310(1), pp. 159-180, 2004.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/CompetitiveAnalysis2004.pdf)
- Online auctions with reusable goods [Hajiaghayi et al. EC-05]
- Reducing truth-telling online mechanisms to online optimization [Awerbuch et al. STOC-03]
- Online learning in online auctions [Blum et al. SODA-03
- Pricing WiFi at Starbucks: Issues in online mechanism design [Friedman & Parkes EC-03]
- [Adaptive limited-supply online auctions Hajiaghayi et al. EC-04](http://www.cs.cmu.edu/~sandholm/cs15-892F15/hajiaghayi04.pdf)
- [Approximately efficient online mechanism design Parkes, Singh and Yanovsky NIPS-04](http://www.cs.cmu.edu/~sandholm/cs15-892F15/mdp_omd04.pdf)
- [An MDP-Based Approach to Online Mechanism Design D. C. Parkes and S. Singh, Proc. NIPS'03, 2003.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/MDP-NIPS03.pdf)
- The price of truth: frugality in truthful mechanisms [Talwar STOC-03]

## RELATED EXCITING TOPICS (which we will probably not have time to cover extensively in class)

### Kidney exchange and exchanges for other organs

- [Dickerson, J. and Sandholm, T. 2015. FutureMatch: Combining Human Value Judgments and Machine Learning to Match in Dynamic Environments. AAAI Conference on Artificial Intelligence . Extended version with appendix.](http://www.cs.cmu.edu/~sandholm/futurematch.aaai15.pdf)
- [Blum, A., Dickerson, J., Haghtalab, N., Procaccia, A., and Sandholm, T. 2015. Ignorance is Almost Bliss: Near-Optimal Stochastic Matching With Few Queries. ACM Conference on Economics and Computation (EC) .](http://www.cs.cmu.edu/~sandholm/ignorance.ec15.pdf)
- [Leishman, R., Stewart, D., Kucheryavaya, A., Callahan, L., Sandholm, T., Aeder, M. 2015. Reasons for Match Offer Refusals and Efforts to Reduce them in the OPTN/UNOS Kidney Paired Donation Pilot Program (KPDPP). American Transplant Congress (ATC) . Slides.](http://www.cs.cmu.edu/~sandholm/ACT2015_Leishman_Reasons%20for%20Match%20Offer%20Refusals%20and%20Efforts.pptx)
- [Dickerson, J. and Sandholm, T. 2014. Multi-Organ Exchange: The Whole is Greater than the Sum of its Parts. AAAI Conference on Artificial Intelligence .](http://www.cs.cmu.edu/~sandholm/multiorgan-aaai2014.pdf)
- [Dickerson, J., Procaccia, A., and Sandholm, T. 2014. Price of Fairness in Kidney Exchange. International Conference on Autonomous Agents and Multiagent Systems (AAMAS).](http://www.cs.cmu.edu/~sandholm/priceOfFairnessInKidneyExchsnge.aamas14.pdf)
- [Dickerson, J. and Sandholm, T. 2014. Balancing Efficiency and Fairness in Dynamic Kidney Exchange. Modern Artificial Intelligence for Health Analytics (MAIHA) workshop at AAAI-14 .](http://www.cs.cmu.edu/~sandholm/balancing%20efficiency%20and%20fairness%20in%20dynamic.maiha14.pdf)
- [Dickerson, J., Procaccia, A., and Sandholm, T. 2014. Empirical Price of Fairness in Failure-Aware Kidney Exchange. Towards Better and more Affordable Healthcare: Incentives, Game Theory, and Artificial Intelligence (HCAGT) workshop at AAMAS-14 .](http://www.cs.cmu.edu/~sandholm/empirical%20price%20of%20fairness%20in%20failure-aware.hcagt14.pdf)
- [Leishman, R., Stewart, D., Monstello, C., Cherikh, W., Sandholm, T., Formica, R., Aeder, M. 2014. The OPTN Kidney Paired Donation Pilot Program (KPDPP): Reaching the Tipping Point in 2013. World Transplant Congress (WTC) . Abstract , presentation .](http://www.cs.cmu.edu/~sandholm/LEISHMAN%20KPD%20general%20update%20(tipping%20point)%20-%20WTC%202014%20(online).doc)
- [Aeder, M., Stewart, D., Leishman, R., Sandholm, T., Formica, R. 2014. Early Outcomes of Transplant Recipients in the OPTN Kidney Paired Donation Pilot Program. World Transplant Congress (WTC) . Abstract , presentation .](http://www.cs.cmu.edu/~sandholm/AEDER%20KPD%20Early%20Outcomes%20-%20WTC%202014%20(online).doc)
- [Stewart, D., Leishman, R., Kucheryavaya, A., Formica, R., Aeder, M., Bingaman, A., Gentry, S., Sandholm, T., and Ashlagi, I. 2014. Exploring the Candidate/Donor Compatibility Matrix to Identify Opportunities to Improve the OPTN KPD Pilot Program's Priority Point Schedule. World Transplant Congress (WTC) . Abstract , poster .](http://www.cs.cmu.edu/~sandholm/STEWART%20KPD%20edgefinder%20analysis%20-%20WTC%202014%20abstract%20(online)%20FINAL%20FULL.doc)
- [A Non-asymptotic Approach to Analyzing Kidney Exchange Graphs. Ding, Y., Ge, D., He, S., and Ryan, C. ACM Conference on Electronic Commerce (EC) , 2015.](http://faculty.chicagobooth.edu/christopher.ryan/research/papers/kidneys.pdf)
- [Design and Analysis of Multi-Hospital Kidney Exchange Mechanisms using Random Graphs. Toulis, P. and Parkes, D. In Games and Economic Behavior , 2015.](http://www.sciencedirect.com/science/article/pii/S0899825615000020)
- [A Dynamic Model of Barter Exchange. Anderson, R., Ashlagi, I., Gamarnik, D., and Kanoria, Y. ACM-SIAM Symposium on Discrete Algorithms (SODA) , 2015. (Long working paper available here .)](http://dl.acm.org/citation.cfm?id=2722129.2722258&coll=DL&dl=ACM&CFID=711378112&CFTOKEN=16227924)
- [Mix and Match: A Strategyproof Mechanism for Multi-hospital Kidney Exchange. Ashlagi, I., Fischer, F., Kash, I., and Procaccia, A. In Games and Economic Behavior , 2015.](http://procaccia.info/papers/mixnmatch.geb.pdf)
- [An Improved 2-agent Kidney Exchange Mechanism. Caragiannis, I., Filos-Ratsikas, A., and Procaccia, A. In Theoretical Computer Science , 2015.](http://www.sciencedirect.com/science/article/pii/S0304397515003205)
- [Dynamic Matching Market Design. Akbarpour, M., Li, S., and Gharan, S. ACM Conference on Electronic Commerce (EC) , 2014. (Long working paper available here .)](http://dl.acm.org/citation.cfm?id=2602887)
- Free Riding and Participation in Large Scale, Multi-hospital Kidney Exchange. Ashlagi, I. and Roth, A. In Theoretical Economics , 2014.
- [Kidney Exchange in Dynamic Sparse Heterogenous Pools. Ashlagi, I., Jaillet, P., and Manshadi, V. ACM Conference on Electronic Commerce (EC) , 2013.](http://arxiv.org/abs/1301.3509)
- [Finding Long Chains in Kidney Exchange using the Traveling Salesman Problem. Anderson, R., Ashlagi, I., Gamarnik, D., and Roth, A. In Proceedings of the National Academy of Sciences , 2015.](http://www.pnas.org/content/112/3/663.abstract)
- [Mechanism Design and Implementation for Lung Exchange. Luo, S. and Tang, P. International Joint Conference on Artificial Intelligence (IJCAI) , 2015.](http://iiis.tsinghua.edu.cn/~kenshin/lung.pdf)
- [Paired and Altruistic Donation in the UK: Algorithms and Experimentation. Manlove, D. and O'Malley, G. In ACM Journal of Experimental Algorithmics , 2014.](http://www.dcs.gla.ac.uk/publications/PAPERS/9383/PAKD-UK-TR.pdf)
- [An Efficient Pricing Algorithm for Clearing Barter Exchanges with Branch-and-Price. Glorie, K., van de Klundert, J., and Wagelmans, A. In Manufacturing & Service Operations Management , 2014.](http://pubsonline.informs.org/doi/abs/10.1287/msom.2014.0496)
- [Egalitarian Pairwise Kidney Exchange: Fast Algorithms via Linear Programming and Parametric Flow. Li, J., Liu, Y., Huang, L., and Tang, P. International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS) , 2014.](http://iiis.tsinghua.edu.cn/~kenshin/kidney.pdf)
- [New Insights on Integer Programming Models for the Kidney Exchange Problem. Constantino, M., Klimentova, X., Viana, A., and Rais, A. In European Journal of Operations Research , 2013.](http://www.sciencedirect.com/science/article/pii/S0377221713004244)
- [Failure-Aware Kidney Exchange. Dickerson, J., Procaccia, A., and Sandholm, T. ACM Conference on Electronic Commerce (EC) , 2013. Dynamic Matching via Weighted Myopia with Application to Kidney Exchange. Dickerson, J., Procaccia, A., and Sandholm, T. 2012. AAAI Conference on Artificial Intelligence (AAAI) , 2012. Optimizing Kidney Exchange with Transplant Chains: Theory and Reality. Dickerson, J., Procaccia, A., and Sandholm, T. International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2012. Online Stochastic Optimization in the Large: Application to Kidney Exchange . Pranjal Awasthi and Tuomas Sandholm. International Joint Conference on Artificial Intelligence (IJCAI), 2009 . ( PDF ) Dynamic Kidney Exchange . Utku Unver. Review of Economic Studies , 2010. A Nonsimultaneous, Extended, Altruistic-Donor Chain. Rees, M., Kopke, J., Pelletier, R., Segev, D., Rutter, M., Fabrega, A., Rogers, J., Pankewycz, O., Hiller, J., Roth, A., Sandholm, T., Ünver, U., and Montgomery, R. New England Journal of Medicine 360(11), 1096-1101, 2009. ( PDF ) Clearing Algorithms for Barter Exchange Markets: Enabling Nationwide Kidney Exchanges. Abraham, D., Blum, A., and Sandholm, T. ACM Conference on Electronic Commerce (EC) , 2007. ( PDF ) Individual Rationality and Participation in Large Scale, Multi-hospital Kidney Exchange . Itai Ashlagi and Alvin E. Roth. EC-11. Al Roth’s Game Theory, Experimental Economics, and Market Design Page . This page has lots of pointers to other matching markets as well.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/failure-aware kidney exchange.ec13.pdf)

### Incentive auctions and holdouts

- [Nguyen, T.and Sandholm, T. 2015. Multi-Option Descending Clock Auction. Draft, August 2015.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/DCA_multi_options.pdf)
- [Nguyen, T.and Sandholm, T. 2014. Optimizing Prices in Descending Clock Auctions. In Proceedings of the ACM Conference on Economics and Computation (EC) . Newer extended version.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/DCA.pdf)
- [Deferred-Acceptance Auctions and Radio Spectrum Reallocation ,”, Paul Milgrom and Ilya Segal, draft, August 2015.](http://www.stanford.edu/~isegal/heuristic.pdf)
- [P. Dütting, V. Gkatzelis, T. Roughgarden, The Performance of Deferred-Acceptance Auctions . In Proceedings of the ACM Conference on Economics and Computation (EC) .](http://paulduetting.com/pubs/ec14-da.pdf)
- Solving the Station Repacking Problem . A. Frechette, N. Newman, K. Leyton-Brown. International Joint Conference on Artificial Intelligence (IJCAI) , 2015.
- [Incentive auction plans at the FCC.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/www.fcc.gov/incentiveauctions)
- [Deferred-Acceptance Heuristic Auctions. Milgrom, P. and Segal, I. Draft, 2013.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/deferred acceptance heuristic auctions.2013.pdf)
- [Concordance among Holdouts. Scott Kommiers and E. Glen Weyl. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/concordance among holdouts.ec11.pdf)

### Combinatorial auctions with "funny money" artificial currency: course allocation as a canonical application

- [Finding Approximate Competitive Equilibria: Efficient and Fair Course Allocation. Othman, A., Budish, E., and Sandholm, T. International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS) , 2010.](http://www.cs.cmu.edu/~aothman/ceei.pdf)
- [Changing the Course Allocation Mechanism at Wharton. Budish, E. and Kessler, J. Working paper, 2015.](http://faculty.chicagobooth.edu/eric.budish/research/ChangingCourseAllocationWharton.pdf)
- [Course Match: A Large-Scale Implementation of Approximate Competitive Equilibrium from Equal Incomes for Combinatorial Allocation. Budish, E., Cachon, G., Kessler, J., and Othman, A. Working paper, 2015.](http://faculty.chicagobooth.edu/eric.budish/research/CourseMatch.pdf)
- [Othman, A., Papadimitriou, C., and Rubinstein, A. 2014. The Complexity of Fairness through Equilibrium. In Proceedings of the ACM Conference on Economics and Computation (EC) . (pdf) .](http://www.cs.cmu.edu/~aothman/equilibrium_fairness.pdf)
- [The Combinatorial Assignment Problem: Approximate Competitive Equilibrium from Equal Incomes. Budish, E. In Journal of Political Economy , 2011.](http://faculty.chicagobooth.edu/eric.budish/research/budish-approxceei-jpe-2011.pdf)

### Prediction markets

- [Nicolas S. Lambert, John Langford, Jennifer Wortman Vaughan, Yiling Chen, Daniel Reeves, Yoav Shoham, and David M. Pennock. An Axiomatic Characterization of Wagering Mechanisms . Journal of Economic Theory .](http://yiling.seas.harvard.edu/wp-content/uploads/JET.pdf)
- [Yiling Chen, Mike Ruberry, and Jennifer Wortman Vaughan. Cost Function Market Makers for Measurable Spaces . EC, 2013. [ Long Version ]](http://dl.acm.org/authorize?6821174)
- [Xi Alice Gao, Andrew Mao, and Yiling Chen. Trick or Treat: Putting Peer Prediction to the Test . Workshop on Crowdsourcing and Online Behavioral Experiments (COBE), in conjunction with ACM EC’13, Philadelphia, PA, June 2013.](http://www.eecs.harvard.edu/econcs/pubs/Gao_acm13.pdf)
- [Jacob Abernethy, Yiling Chen, and Jennifer Wortman Vaughan. Efficient Market Making via Convex Optimization, and a Connection to Online Learning . ACM Transactions on Economics and Computation . Vol. 1, no. 2, pp. 12:1-12:39, May 2013](http://dl.acm.org/authorize?6820440)
- [Othman, A., Pennock, D., Reeves, D., and Sandholm, T. 2013. A Practical Liquidity-Sensitive Automated Market Maker. ACM Transaction on Economics and Computation (TEAC) , to appear. (Conference version in EC-10.) Othman, A. and Sandholm, T. 2013. The Gates Hillman prediction market. Review of Economic Design , 17(2), 95-128. (Conference version "Automated Market-Making in the Large: The Gates Hillman Prediction Market" in EC-10.) Othman, A. and Sandholm, T. 2012. Profit-Charging Market Makers with Bounded Loss, Vanishing Bid/Ask Spreads, and Unlimited Market Depth. In Proceedings of the ACM Conference on Electronic Commerce (EC) . Othman, A. and Sandholm, T. 2012. Rational Market Making with Probabilistic Knowledge. In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS) . Othman, A. and Sandholm, T. 2011. Inventory-based versus Prior-based Options Trading Agents. Algorithmic Finance 1:95-121. Othman, A. and Sandholm, T. 2011. Liquidity-Sensitive Automated Market Makers via Homogeneous Risk Measures. Workshop on Internet And Network Economics (WINE). An Efficient Monte-Carlo Algorithm for Pricing Combinatorial Prediction Markets for Tournaments. Lirong Xia and David M. Pennock. IJCAI-11. An Optimization-Based Framework for Automated Market-Making. Jacob Abernethy , Yi ling Chen, and Jennifer Wort man Vaughan. EC-11. Only Valuable Experts Can Be Valued. Moshe Babaioff, Liad Blumrosen, Nicolas Lambert, and Omer Reingold. EC-11. Othman, A. and Sandholm, T. 2011. Automated Market Makers That Enable New Settings: Extending Constant-Utility Cost Functions. In Proceedings of the Conference on Auctions, Market Mechanisms and Their Applications (AMMA) . Othman, A. and Sandholm, T. 2010. Decision Rules and Decision Markets. In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS) . Othman, A. and Sandholm, T. 2010. When Do Markets with Simple Agents Fail? In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS) . Computational Aspects of Prediction Markets, Chapter 26 of Algorithmic Game Theory .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/liquidity-sensitive automated market maker.teac.pdf)

### Bundling

- [Kroer, C. and Sandholm, T. 2015. Computational Bundling for Auctions. In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS) . Also: Computational Bundling for Auctions, CMU Computer Science Department Technical Report CMU-CS-13-111, 2013.](http://www.cs.cmu.edu/~sandholm/computationalbundling.aamas15.fromACM.pdf)
- [Signaling schemes for revenue maximization. Emek, Y., Feldman, M., Gamzu, I., Paes Leme, R., and Tennenholtz, M. EC-12.](http://delivery.acm.org/10.1145/2230000/2229051/p514-emek.pdf?ip=128.2.211.42&id=2229051&acc=ACTIVE%20SERVICE&key=C2716FEBFA981EF1D8ED4F16102DA82BADDD11EBB600FF97&CFID=244309932&CFTOKEN=69711336&__acm__=1378860426_b806073d6cb994dc168068fa29dd8787)
- [Send Mixed Signals – Earn More, Work Less. Miltersen, P., and Sheffet, O. EC-12.](http://delivery.acm.org/10.1145/2230000/2229033/p234-miltersen.pdf?ip=128.2.211.42&id=2229033&acc=ACTIVE%20SERVICE&key=C2716FEBFA981EF1D8ED4F16102DA82BADDD11EBB600FF97&CFID=244309932&CFTOKEN=69711336&__acm__=1378860606_d21f9953205d2320717118872e45d610)
- [Revenue Maximization via Hiding Item Attributes. Guo, M. and Deligkas, A. IJCAI-13.](http://www.cs.cmu.edu/~sandholm/http://cgi.csc.liv.ac.uk/~mingyu/publication/ijcai13.pdf)
- [Automated Channel Abstraction for Advertising Auctions. Walsh, W., Boutilier, C., Sandholm, T., Shields, R., Nemhauser, G., and Parkes, D. In Proceedings of the Ad Auctions Workshop, 2009 . ( PDF )](http://www.cs.cmu.edu/~sandholm/channel_abstraction.aaai10.pdf)

### Externalities

- [Money for Nothing: Exploiting Negative Externalities . Changrong Deng and Saša Peke c. EC -11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/money for nothing - exploiting negative externalities.ec11.pdf)
- [Krysta, P., Michalak, T., Sandholm, T., and Wooldridge, M. 2010. Combinatorial Auctions with Externalities. Extended version of AAMAS-10 paper.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/combinatorial auctions with externalities.draft 2011.pdf)
- [Conitzer, V. and Sandholm, T. Computing Optimal Outcomes under an Expressive Representation of Settings with Externalities. Journal of Computer and System Sciences (JCSS) , special issue on Knowledge Representation and Reasoning, to appear. Early version in AAAI-05 .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/computingOutcomesUnderExpressiveExternalitiesWithVince.JCSS.pdf)
- Conitzer, V. and Sandholm, T. 2011. Expressive Markets for Donating to Charities. Artificial Intelligence , 175(7-8), 1251-1271, special issue on Representing, Processing, and Learning Preferences: Theoretical and Practical Challenges. Early version in EC-04.
- [Boutilier, C., Parkes, D., Sandholm, T., and Walsh, W. 2008. Expressive Banner Ad Auctions and Model-Based Online Optimization for Clearing. In Proceedings of the National Conference on Artificial Intelligence (AAAI) . ( PDF )](http://www.cs.cmu.edu/~sandholm/expressiveBannerAdAuctions.AAAI08.pdf)
- [Optimize-and-Dispatch Architecture for Expressive Ad Auctions by David C. Parkes and Tuomas Sandholm. In the Proceedings of First Workshop on Sponsored Search Auctions, 2005.( PDF )](http://www.cs.cmu.edu/~sandholm/optimize-and-dispatch.ws05.pdf)
- [Externalities in Keyword Auctions: An Empirical and Theoretical Assessment , R. Gomes, N. Immorlica and E. Markakis, in Proc. 5th Workshop on Ad Auctions (2009). ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/externalities09.pdf)
- [On Expressing Value Externalities in Position Auctions . Florin Constantin, Malvika Rao, Chien-Chung Huang, and David C. Parkes. AAAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ExpressingExternalitiesInPositionAuctions.aaai11.pdf)

### Advertising markets

#### Sponsored search

- [Methodology for Designing Reasonably Expressive Mechanisms with Application to Ad Auctions. Benisch, M., Sadeh, N., and Sandholm, T. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) . ( PDF )](http://www.cs.cmu.edu/~sandholm/methodologyReasonablyExpressive.ijcai09.pdf)
- [Multi-Keyword Sponsored Search. Peerapong Dhangwatnotai. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/multi-keyword sponsored search.ec11.pdf)
- [Reserve Prices in Internet Advertising Auctions: A Field Experiment. Michael Ostrovsky and Michael Schwarz. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/reserve prices in ad auctions.ec11.pdf)
- [Machine Learning in an Auction Environment . International Conference on the World Wide Web (WWW), 2014. Hummel, P. and McAfee, P.](http://vita.mcafee.cc/PDF/VOL2.pdf)
- [On Expressing Value Externalities in Position Auctions . Florin Constantin, Malvika Rao, Chien-Chung Huang, and David C. Parkes. AAAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ExpressingExternalitiesInPositionAuctions.aaai11.pdf)
- [Computational analysis of perfect-information position auctions , D. Thompson and K. Leyton-Brown, in Proc. ACM EC'09, 2009. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/AdAuctionEqEC09.pdf)
- [Position Auctions with Budgets: Existence and Uniqueness . By Itai Ashlagi, Mark Braverman, Avinatan Hassidim, Ron Lavi, and Moshe Tennenholtz.Draft 2009. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/position-auctions-with-budgets09.pdf)
- [Is Efficiency Expensive? Roughgarden, T. and Sundararajan, M. 3rd Workshop on Sponsored Search, 2007. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/EfficiencyExpensive07.pdf)
- [H. Varian. Position Auctions To appear in International Journal of Industrial Organization. (A theoretical and empirical analysis of the search keyword auction used by Google and Yahoo.) ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/position.pdf)
- [Internet Advertising and the Generalized Second Price Auction: Selling Billions of Dollars Worth of Keywords . By M. Schwartz, B. Edelman and M. Ostrovsky. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/gsp051003.pdf)
- [Optimize-and-Dispatch Architecture for Expressive Ad Auctions by David C. Parkes and Tuomas Sandholm. In the Proceedings of First Workshop on Sponsored Search Auctions, 2005. ( PDF )](http://www.cs.cmu.edu/~sandholm/optimize-and-dispatch.ws05.pdf)
- [J. Tomlin, Z. Abrams and O. Mendelevitch. Optimal delivery of sponsored search advertisements subject to budget constraints . In Proc. ACM Conference on Electronic Commerce (EC'07) , pp. 272-278, 2007. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/OptimalDeliveryEC07.pdf)
- [Externalities in Keyword Auctions: An Empirical and Theoretical Assessment , R. Gomes, N. Immorlica and E. Markakis, in Proc. 5th Workshop on Ad Auctions (2009). ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/externalities09.pdf)
- [Sponsored search auctions with Markovian users , G. Aggarwal, J. Feldman, S. Muthukrishnan, and M. Pal. In Proc. 4th Workshop on Ad Auctions (2008). ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/SponsoredSearchAuctions08.pdf)
- [Sponsored Search with Contexts. Eyal Even-Dar, Michael Kearns, and Jennifer Wortman. WWW-07.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/sponsored search with contexts.www07.pdf)
- [Bid Optimization for Broad Match Ad Auctions. Eyal Even-Dar, Yishay Mansour, Vahab S. Mirrokni, S. Muthukrishnan, Uri Nadav. WWW-09.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/bid optimization for broad match.www09.pdf)
- [Sponsored Search Auctions. Lahaie, S., Pennock, D., Saberi, A., and Vohra, R. Chapter 28 of the book Algorithmic Game Theory , 2007. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/algorithmic-game-theory.pdf)

#### Display advertising "exchanges", i.e., spot markets for remnant inventory where selling is one impression at a time

- [When Does Improved Targeting Increase Revenue? International Conference on the World Wide Web (WWW), 2015 . Hummel, P. and McAfee, P.](http://vita.mcafee.cc/PDF/targeting.pdf)
- [Display Advertising Auctions with Arbitrage . Transactions in Economics and Computation , to appear. Cavallo, R, McAfee, P., and Vassilvitskii, S.](http://vita.mcafee.cc/PDF/Arbitrage.pdf)
- [To Match or Not to Match: Economics of Cookie Matching in Online Advertising , Transactions in Economics and Computation , To Appear (with Arpita Ghosh, Mohammad Mahdian and Sergei Vassilvitskii).](http://vita.mcafee.cc/PDF/CookieMatching2.pdf)
- [Signaling schemes for revenue maximization. Emek, Y., Feldman, M., Gamzu, I., Paes Leme, R., and Tennenholtz, M. EC-12. Send Mixed Signals – Earn More, Work Less. Miltersen, P., and Sheffet, O. EC-12. Revenue Maximization via Hiding Item Attributes. Guo, M. and Deligkas, A. IJCAI-13.](http://delivery.acm.org/10.1145/2230000/2229051/p514-emek.pdf?ip=128.2.211.42&id=2229051&acc=ACTIVE%20SERVICE&key=C2716FEBFA981EF1D8ED4F16102DA82BADDD11EBB600FF97&CFID=244309932&CFTOKEN=69711336&__acm__=1378860426_b806073d6cb994dc168068fa29dd8787)
- [Yield Optimization of Display Advertising with Ad Exchange. Santiago Balseiro, Jon Feldman, Vahab Mirrokni, and S. Muthukrishnan. Extended version of EC-11 paper.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/yield optimization for display ad exchange.extended version of EC-11.pdf)

#### Display advertising for premium inventory -- typically sold manually in campaigns but automatically dispatched

- [Automated Channel Abstraction for Advertising Auctions. Walsh, W., Boutilier, C., Sandholm, T., Shields, R., Nemhauser, G., and Parkes, D. In Proceedings of the Ad Auctions Workshop, 2009 . ( PDF )](http://www.cs.cmu.edu/~sandholm/channel_abstraction.aaai10.pdf)
- [Near Optimal Online Algorithms and Fast Approximation Algorithms for Resource Allocation Problems. Nikhil Devanur, Kamal Jain, Balasubramanian Sivan, and Christopher A. Wilkens. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/near optimal online algs for resource allocation.ec11.pdf)
- [Boutilier, C., Parkes, D., Sandholm, T., and Walsh, W. 2008. Expressive Banner Ad Auctions and Model-Based Online Optimization for Clearing. In Proceedings of the National Conference on Artificial Intelligence (AAAI) . ( PDF )](http://www.cs.cmu.edu/~sandholm/expressiveBannerAdAuctions.AAAI08.pdf)
- [Optimize-and-Dispatch Architecture for Expressive Ad Auctions by David C. Parkes and Tuomas Sandholm. In the Proceedings of First Workshop on Sponsored Search Auctions, 2005.( PDF )](http://www.cs.cmu.edu/~sandholm/optimize-and-dispatch.ws05.pdf)

#### TV advertising, print ads, etc.

- [Google's auction for TV ads , N. Nisan et al., ICALPS 2009. ( PDF ). (This auction has been discontinued.)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/GoogleTV-ICALPS09.pdf)
- [Pricing guidance in ad sale negotiations: The PrintAds example , A. Juda, S. Muthukrishnan and A. Ratogi, in 3rd. Int. W. on Data Mining and Audience Intelligence for Advertising, 2009. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/PricingGuidancePrintAds2009.pdf)

#### Core-selecting combinatorial auctions

- [New Core-Selecting Payment Rules with Better Fairness and Incentive Properties. Benjamin Lubin, Benedikt Bünz, and Sven Seuken. August 2015. [pdf]](http://www.ifi.uzh.ch/ce/publications/Fairness_and_Incentives.pdf)
- [A Faster Core Constraint Generation Algorithm for Combinatorial Auctions. Benedikt Bünz, Sven Seuken, and Benjamin Lubin. AAAI Conference on Artificial Intelligence (AAAI) , 2015. [pdf]](http://www.ifi.uzh.ch/ce/publications/A_Faster_CCG_Algorithm_Buenz_et_al_AAAI_2015.pdf)
- [Envy Quotes and the Iterated Core-Selecting Combinatorial Auction. Abe Othman and Tuomas Sandholm. In Proceedings of the National Conference on Artificial Intelligence (AAAI), 2010 .](http://www.cs.cmu.edu/~sandholm/cs15-892F15/envy quotes for CAs.aaai10.pdf)
- [Core-Selecting Package Auctions . Robert Day and Paul Milgrom. International Journal of Game Theory , 36, 2008, 393-407](http://www.stanford.edu/~milgrom/publishedarticles/Core%20Selecting%20Package%20Auctions.pdf)
- [Optimal Incentives in Core-Selecting Auctions. Robert Day and Paul Milgrom, 8/25/2010. In the Handbook of Market Design , Oxford University Press.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/www.milgrom.net/publications/downloads/Incentives%20in%20Core-Selecting%20Auctions%2008-25-2010.pdf)
- [Fair Payments for Efficient Allocations in Public Sector Combinatorial Auctions. Robert Day and S. Raghavan. M anagement Science. 53 (9), September 2007, pp. 1389-1406. PDF .](http://users.business.uconn.edu/bday/FPEA.pdf)
- [Core-Selecting Auctions with Incomplete Information. Lawrence M. Ausubel and Oleg V. Baranov. Draft, August 2010.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/ausubel-baranov-core-selecting-auctions-with-incomplete-info.2010.pdf)

#### Dynamically auctioning wireless spectrum (by the way, Google advocates dynamic provisioning to the FCC, and there is talk of such in Ireland as well)

- A General Framework for Clearing Auction of Wireless Spectrum. Sorabh Gandhi, Chiranjeeb Buragohain, Lili Cao, Haitao Zheng and Subhash Suri. IEEE DySPAN'07, 2007.

#### Distributed implementation

- [MDPOP: Faithful Distributed Implementation of Efficient Social Choice Problems by Adrian Petcu, Boi Faltings, and David C. Parkes. In the Proc. 5th International Joint Conference on Autonomous Agents and Multiagent Systems(AAMAS), pages 1397-1404, 2006.( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/MDPOP-AAMAS06.pdf)
- [Specification Faithfulness in Networks with Rational Nodes by Jeffrey Shneidman and David C. Parkes. In the Proc. 23rd ACM Symp. on Principles of Distributed Computing (PODC), St. John's, Canada, pages 88-97, 2004.( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/SpecificationFaithfulnessPODC04.pdf)
- [Distributed computing meets game theory: robust mechanisms for rational secret sharing and multiparty computation , Proceedings of the Twenty-Fifth Annual ACM Symposium on Principles of Distributed Computing , 2006, pp. 53-62 (J. Halpern, I. Abraham, D. Dolev, and R. Gonen). ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/DCmeetsGT-PODC06.pdf)
- [Rational secret sharing and multiparty computation , Proceedings of 36th ACM Symposium on Theory of Computing , 2004, pp. 623-632 (J. Halpern and V. Teague). ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/RationalSecretSharingSTOC04.pdf)
- [J. Feigenbaum, C. Papadimitriou, R. Sami, and S. Shenker, A BGP-based Mechanism for Lowest-Cost Routing , Distributed Computing 18 (2005), pp. 61-72. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/BGP-MechanismLowestCostDC05.pdf)
- [J. Feigenbaum, M. Schapira, and S. Shenker, Distributed Algorithmic Mechanism Design , to appear in Algorithmic Game Theory , Cambridge University Press, 2007. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Rahul-thesis.pdf)

#### Privacy in mechanism design

- [Selling Privacy at Auction . Arpita Ghosh and Aaron Roth. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/selling privacy at auction.ec11.pdf)
- [Efficiency and Privacy Tradeoffs in Mechanism Design. Xin Sui and Craig Boutilier. AAAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/efficiency-privacy tradeoffs in MD.aaai11.pdf)
- [Brandt, F. and Sandholm, T. 2008. On the Existence of Unconditionally Privacy-Preserving Auction Protocols. ACM Transactions on Information and System Security .( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/file://///afs/cs.cmu.edu/user/sandholm/www/privateauctions_journal.pdf)
- Brandt, F. and Sandholm, T. 2005. Unconditional Privacy in Social Choice. In Proceedings of the Theoretical Aspects of Reasoning about Knowledge (TARK) conference.
- [Brandt, F. and Sandholm, T. Efficient Privacy-Preserving Protocols for Multi-Unit Auctions. International Conference on Financial Cryptography and Data Security (FC) , LNCS 3570. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/file://///afs/cs.cmu.edu/user/sandholm/www/privacy_preserving_multi-unit_auctions.lncs05.pdf)
- [Brandt, F. and Sandholm, T. 2005. Decentralized Voting with Unconditional Privacy. AAMAS-05.( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/file://///afs/cs.cmu.edu/user/sandholm/www/decentralized_voting.aamas05.pdf)
- [Brandt, F. and Sandholm, T. 2004. On Correctness and Privacy in Distributed Mechanisms In Proceedings of the Agent-Mediated Electronic Commerce(AMEC) workshop, Springer LNAI 3937. ( PDF )](http://www.cs.cmu.edu/~sandholm/correctness_and_privacy.amec04LNAI.pdf)
- Sergei Izmalkov, Matt Lepinski and Silvio Micali. 2005. Rational Secure Computation and Ideal Mechanism Design . FOCS. Relies on a ballot box.
- [Practical Secrecy-Preserving, Verifiably Correct and Trustworthy Auctions by David C. Parkes, Michael O. Rabin, Stuart M. Shieber, and Christopher Thorpe., In Electronic Commerce Research and Applications, 2007, to appear. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Secrecy-PreservingECRA07.pdf)
- Cryptographic Securities Exchanges by Christopher Thorpe and David C. Parkes. In Proc. International Conference on Financial Cryptography and Data Security , 2007.

#### Exotic contract types (�It�s not the figures lying, it�s the liars figuring� -- Mark Twain)

- [Leveled commitment contracts and strategic breach , Sandholm, T. and Lesser, V., Games and Economic Behavior , 2001 ( PDF )](http://www.cs.cmu.edu/~sandholm/leveled.geb.pdf)
- [Surplus Equivalence of Leveled Commitment Contracts. Sandholm, T. and Zhou, Y., Artificial Intelligence 142, 239-264, 2002. ( PDF )](http://www.cs.cmu.edu/~sandholm/surplus_equivalence.aij.pdf)
- [Algorithms for optimizing leveled commitment contracts Sandholm-Sikka-Norden IJCAI-99 ( PS )](http://www.cs.cmu.edu/~sandholm/algs.ijcai99.ps)
- [Efficient Mechanisms with Risky Participation. Cavallo, R.. IJCAI-11. (Could this be applied to make leveled commitment contracts work without knowledge of possible futures?)](http://www.cs.cmu.edu/~sandholm/cs15-892F15/efficient mechanisms with risky participation.ijcai11.pdf)

#### Coalition formation

- [Rahwan, T., Michalak, T., Wooldridge, M., Jennings, N. Coalition structure generation: A survey. Artificial Intelligence 229: 139–174, 2015.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Coalition Structure Generation Survey (AIJ-2015).pdf)
- [Minimum Search To Establish Worst-Case Guarantees in Coalition Structure Generation. Talal Rahwan, Tomasz Michalak, and Nicholas R. Jennings. IJCAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/minimum coalition structure search.ijcai11.pdf)
- [Coalition structure generation with worst case guarantees . Sandholm et al AIJ-99 ( PDF )](http://www.cs.cmu.edu/~sandholm/coalstruct.aij.pdf)
- [Computing Shapley values, manipulating value division schemes, and checking core membership in multi-issue domains. In Proceedings of the National Conference on Artificial Intelligence (AAAI), pp. 219-225, 2004. Conitzer and Sandholm.( PDF )](http://www.cs.cmu.edu/~sandholm/computing_shapley.aaai04.pdf)
- [Marginal Contribution Nets: A Compact Representation Scheme for Coalitional Games . S. Ieong, Y. Shoham., EC-05. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/MarginalContributionEC05.pdf)
- [Complexity of Constructing Solutions in the Core Based on Synergies Among Coalitions. Artificial Intelligence , 170: 607-619, 2006, Conitzer and Sandholm. ( PDF )](http://www-2.cs.cmu.edu/~sandholm/core_complexity.AIJ.pdf)
- [Coalition formation among agents whose computation is costly. Sandholm & Lesser AIJ-97 ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/sand95.pdf)
- [Sharing the cost of multicast transmissions . Feigenbaum et al. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/MulticastTransmissions.pdf)
- Coalition-proof implementation via LP duality . Vazirani et al
- [Yokoo, M., Conitzer, V., Sandholm, T., Ohta., N., and Iwasaki, A., 2005. Coalitional Games in Open Anonymous Environments. In Proceedings of the National Conference on Artificial Intelligence (AAAI) , Pittsburgh, PA. ( PDF )](http://www.cs.cmu.edu/~sandholm/coalitional_anonymous.aaai05.pdf)
- [Ohta, N., Iwasaki, A., Yokoo, M., Maruono, K., Conitzer, V., and Sandholm, T., 2006. A Compact Representation Scheme for Coalitional Games in Open Anonymous Environments. In Proceedings of the National Conference on Artificial Intelligence (AAAI) . ( PDF )](http://www-2.cs.cmu.edu/~sandholm/compactcoalitional.aaai06.pdf)

#### Social networks and multi-level marketing

- [Mechanisms for Multi-Level Marketing. Yuval Emel, Ron Kardi, Moshe Tennenholtz, and Aviv Zohar. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/mechanisms for multi-level marketing.ec11.pdf)

#### Safe exchange

- [Incentive-Compatible Escrow Mechanisms. Jens Witkowski, Sven Seuken, and David C. Parkes. AAAI-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/incentive-compatible escrow mechanisms.aaai11.pdf)
- [Sandholm, T. and Wang, X. 2002. (Im)possibility of Safe Exchange Mechanism Design. National Conference on Artificial Intelligence (AAAI). ( PDF )](http://www-2.cs.cmu.edu/~sandholm/Impossibility.aaai02.pdf)
- [Safe exchange planner , Sandholm-Ferrandon ICMAS-00 ( PDF )](http://www.cs.cmu.edu/~sandholm/sep.icmas00_submission.pdf)
- [Defection-free exchange mechanisms for information goods Yokoo ICMAS-00 ( IEEE Link )](http://www.computer.org/portal/web/csdl/doi/10.1109/ICMAS.2000.858452)
- Cryptographic safe exchange techniques
- Automated escrow services

#### Best-response auctions

- [Best-Response Auctions. Noam Nisan, Michael Schapira, Gregory Valiant, and Aviv Zo har. EC-11.](http://www.cs.cmu.edu/~sandholm/cs15-892F15/best-response auctions.ec11.pdf)

#### Reputation systems (these systems are prevalent - e.g., eBay - but they are all manipulable)

- ["Manipulation-Resistant Reputation Systems", by Friedman/Resnick/Sami., Chapter 27 of the book Algorithmic Game Theory , Nisan, Roughgarden, Tardos, and Vazirani (eds.),Cambridge University Press, 2007. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/algorithmic-game-theory.pdf)
- [R. Jurca and B. Faltings. Collusion Resistant, Incentive Compatible Feedback Payments . Proceedings of the ACM Conference on E-Commerce (EC'07) , pp. 200-209, 2007. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/p200-jurca.pdf)
- [R. Jurca and B. Faltings. Minimum Payments that Reward Honest Reputation Feedback . Proceedings of the ACM Conference on Electronic Commerce (EC2006) , pp.190-199, 2006. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/p190-jurca.pdf)
- "Combinatorial auctions with false-name bidders" (Yokoo�s chapter in the MIT Press book Combinatorial Auctions , 2006)

#### Cost sharing

- [Mehta, Roughgarden, and Sundararajan, Beyond Moulin Mechanisms , EC '07. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/BeyondMoulinEC07.pdf)

#### Worst-case Nash equilibrium (what’s the price of anarchy?)

- [Vasilis Syrgkanis, Eva Tardos. Composable and Efficient Mechanisms , Symposium on the Theory of Computing, STOC'13.](http://arxiv.org/pdf/1211.1325.pdf)
- [Entire section on this in the book Algorithmic Game Theory ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/algorithmic-game-theory.pdf)
- [How bad is selfish routing? Roughgarten & Tardos, 2001. ( PDF )](http://www.cs.cmu.edu/~sandholm/cs15-892F15/Selfish routing.pdf)

## Other resources

- [Tuomas Sandholm�s home page](http://www.cs.cmu.edu/~sandholm/)
- [Al Roth�s Market Design Ideas Page](http://marketdesigner.blogspot.com/)

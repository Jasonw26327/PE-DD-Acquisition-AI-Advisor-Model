# The 20 real transactions

Every transaction below is verified against a public source, and each carries buyer, target, seller or sponsor, and the exit route in `data/corpus_review.xlsx`. Feature values (rights, consent, regime) are an analyst reading of the public record rather than facts from it, and are marked for sign-off.

Generated rows carry no counterparties, since they describe no real transaction. Their remarks column instead explains in plain language why the rule produced that decision, so a reviewer can sign a row off without reading the code.

## Strategic buyers paying for data

**Microsoft / GitHub** (June 2018, $7.5B, all stock). Sold by GitHub's founders and venture holders including Andreessen Horowitz, Sequoia, Thrive Capital and Institutional Venture Partners. Funded through an incremental share repurchase, no cash component. The asset was the public code corpus and the developer relationship, not the revenue. The rights lesson: code is publicly licensed but licences vary per repository, so a blanket training right never existed. GitHub later became the substrate for Copilot and the subject of litigation over training on licensed code. Public availability is not permission.

**Salesforce / Tableau** (June 2019, $15.7B, all stock). Pitched as completing Customer 360 and feeding Einstein. What a data buyer would underwrite is the analytical content customers created inside Tableau, which is customer-owned and carries no training right established by the deal filing. Acquiring the platform does not acquire the data on the platform.

**Twilio / Segment** (announced Oct 2020, closed Nov 2020, $3.2B, mostly stock). Sold by venture holders including Accel, GV and Thrive. Segment holds first-party behavioural data on behalf of thousands of companies: exactly what a model buyer wants, and exactly what CDP contracts forbid using beyond the customer's own purposes. Often misquoted at $320M; the figure is $3.2 billion.

**Okta / Auth0** (announced Mar 2021, closed May 2021, $6.5B, all stock). Sold by Bessemer, Meritech and Salesforce Ventures at roughly ten times Auth0's last private valuation. Authentication telemetry at scale has obvious fraud-modelling value and no established lawful basis for secondary use. Attractive asset, unresolved rights, which should produce an abstention rather than a discount.

**Microsoft / Nuance Communications** (announced Apr 2021, closed Mar 2022, $19.7B cash). Microsoft's second-largest acquisition after LinkedIn. The asset is clinical documentation: physician dictation and ambient notes across a large share of US hospitals. The archetype of a high-value, high-restriction asset. Records are protected health information, so a training basis must be established provider by provider under HIPAA business-associate terms.

**Oracle / Cerner** (announced Dec 2021, closed June 2022, $28.3B cash tender). Oracle's largest acquisition, for one of the two dominant US electronic health record vendors. Longitudinal patient records held on behalf of health systems. Every meaningful use runs through HIPAA, state law and provider governance, none of which transfers with the shares. Pairs with Nuance as the cases where the asset is most valuable and the rights least transferable.

**Thomson Reuters / Casetext** (announced June 2023, closed Aug 2023, $650M cash). Sold by Union Square Ventures, Canvas Ventures and Touchdown Ventures. About 100 employees, more than 10,000 law firm and legal department customers, and CoCounsel, a GPT-4 based assistant. The largest legaltech deal on record at the time. The rights position is comparatively clean because much of the underlying material is public case law, which is why it commanded a full price rather than a rights discount.

**Meta / Scale AI** (June 2025, $14.3B for 49% non-voting, valuing Scale at ~$29B). Founder Alexandr Wang moved to Meta's superintelligence lab; Jason Droege became CEO. The purest expression of the thesis that data is the asset. The structure matters as much as the price: a non-voting minority secures access and talent while avoiding a merger review, and left Scale free to keep serving other labs, several of which reduced their business afterwards. Model companies will pay extraordinary multiples for annotation capacity and will structure creatively to get it.

## Sponsor deals, and the sponsor-to-strategic route

**Vista Equity Partners / Marketo** (Aug 2016, ~$1.79B take-private) and **Adobe / Marketo** (Sept 2018, $4.75B cash). The clearest complete story in the set, and the reason both ends are in the corpus. Vista took a listed marketing automation vendor private during a soft period for SaaS multiples, held it two years, and sold to Adobe at roughly 2.7 times entry. For a fund holding a similar asset this is the template: the strategic pays for the data and the customer relationships, the sponsor captures the difference. It also shows why rights hygiene during the hold period converts into price at exit.

**Francisco Partners and TPG / New Relic** (announced July 2023, $6.5B all-cash take-private at $87 per share, a 26% premium). Read this against Cisco and Splunk in the same window: two similar observability data assets, one bought by a strategic at $28B and one by sponsors at $6.5B. The difference is not the data, it is who can use it. A sponsor underwrites cash flow; a strategic can underwrite the data because it has somewhere to put it. That gap is the arbitrage the AI-buyer thesis describes.

**Thoma Bravo / Coupa Software** (announced Dec 2022, closed Feb 2023, $8B, with Abu Dhabi Investment Authority co-investing). Coupa's community spend data, aggregated purchasing behaviour across thousands of buyers and suppliers, is a genuine data asset with benchmarking value. It is contributed under agreements written for benchmarking, not model training, which decides whether an AI-buyer exit is available later or has to be negotiated for first.

**Thoma Bravo / ForgeRock** (announced Oct 2022, closed Aug 2023, $2.3B at $23.25 per share, later merged with Ping Identity). Closed only after an extended US antitrust review, because Thoma Bravo already owned Ping. Two lessons: sponsor consolidation inside a sector draws the same regulatory attention a strategic deal would, and merging two identity data estates raises a rights question neither company faced alone, since consents were given to one vendor rather than to the combined entity.

## Carve-outs

**Broadcom / Symantec enterprise security business** (Aug 2019, $10.7B cash). Broadcom took the enterprise division; the remainder renamed itself NortonLifeLock and kept the consumer business. Threat telemetry drawn from customer environments is valuable for security modelling and tightly restricted by the contracts it arrives under. Carve-outs make rights harder, not easier: contracts and consents were written for an entity that no longer holds the asset and often need novation.

## Deals where the data lens gives the wrong answer

**NVIDIA / Arm** (proposed Sept 2020 at ~$40B, terminated Feb 2022; seller SoftBank Group and the Vision Fund). Abandoned after the FTC sued and the UK CMA and EU opened in-depth reviews. The objection was competition and Arm's neutrality as an IP licensor to the whole industry, not data rights. NVIDIA wrote off a $1.36B prepayment; SoftBank listed Arm on Nasdaq in 2023 instead. A model trained only on rights features will get this case wrong.

**Meta / Giphy** (May 2020, ~$315-400M; divested May 2023 for $53M to Shutterstock). The UK CMA found the merger anticompetitive and ordered divestiture. Meta recovered roughly 13% of what it paid. No rights diligence would have caught this, because the binding constraint was competition law in a jurisdiction representing a minority of the user base. Kept specifically because it marks a limit on the whole method.

**IBM / Red Hat** (announced Oct 2018, closed July 2019, $34B cash at $190 per share). IBM's largest acquisition. What was bought was a distribution and support relationship over software that is, by licence, freely available. Essentially no proprietary data moat; the thesis was hybrid cloud position. An example where the correct action is a conventional strategic rationale and the data lens adds nothing.

**Databricks / MosaicML** (June 2023, ~$1.3B, largely stock; sold by Lux Capital and DCVC). Roughly 60 people, valued at $222M a year earlier. What was bought was training infrastructure and the team, not a corpus. Not every deal in this market is a data deal, and a model that treats every AI acquisition as one will misprice the capability and talent cases.

## The cautionary one

**Salesforce / Kustomer** (announced Nov 2020, closed 2021, ~$1.35B; sold back 2023 at a reported valuation near $250M to a group led by Battery Ventures and Redpoint, its former backers). The stated asset was billions of customer service conversations, a strong training corpus on paper. In practice the conversations belong to the end customers, integration stalled, and the thesis did not survive contact with the contracts. Conversation volume is not the same as usable conversation data.

**Cisco / Splunk** (announced Sept 2023, closed Mar 2024, $28B cash at $157 per share, a 31% premium; Starboard Value and Hellman & Friedman held positions beforehand). Cisco's largest acquisition in four decades. Machine and log data held on behalf of enterprise customers, so the training-rights question sits in thousands of customer contracts rather than the merger agreement. Included partly as a correction: this deal is sometimes misattributed to a private equity buyer at around $6.5B, a figure that appears to be borrowed from the New Relic take-private in the same period.

## Status

These 20 records sit in the `anchor_eval` split and are never trained on.

Eleven of the twenty anchors resolve to ABSTAIN under the current rule: Tableau, Auth0, Giphy, Arm, Nuance, Cerner, Scale AI, Coupa, ForgeRock, MosaicML and Kustomer. That is a high share, and worth a decision when you review the sheet rather than accepting it by default. Two of them look like genuine rule problems rather than genuine abstentions. MosaicML abstains only because its rights position is press-sourced, which is true but beside the point in a deal that bought a team rather than a corpus. Coupa abstains on the PCI branch, which may be too blunt if the spend data is tokenised. Where the rule is wrong, the fix belongs in `decide()` rather than in the record. These 20 transactions are the fastest way to find those places.

## Sources

- [Microsoft / GitHub](https://news.microsoft.com/source/2018/06/04/microsoft-to-acquire-github-for-7-5-billion/)
- [Twilio / Segment completion](https://www.twilio.com/en-us/press/releases/twilio-completes-acquisition-segment-market-leading-customer-data-platform)
- [Cisco / Splunk completion](https://investor.cisco.com/news/news-details/2024/Cisco-Completes-Acquisition-of-Splunk/default.aspx)
- [NVIDIA / Arm termination](https://nvidianews.nvidia.com/news/nvidia-and-softbank-group-announce-termination-of-nvidias-acquisition-of-arm-limited) and the [FTC statement](https://www.ftc.gov/news-events/news/press-releases/2022/02/statement-regarding-termination-nvidia-corps-attempted-acquisition-arm-ltd)
- [Meta sells Giphy to Shutterstock for $53M](https://www.cnbc.com/2023/05/23/meta-sells-giphy-to-shutterstock-at-a-loss-in-a-53-million-deal.html)
- [Microsoft closes Nuance at $19.7B](https://aibusiness.com/companies/microsoft-closes-its-19-7-billion-acquisition-of-nuance)
- [Oracle closes Cerner at $28.3B](https://www.healthcaredive.com/news/oracle-closes-283b-buy-huge-growth-engine-cerner/625103/)
- [Thomson Reuters completes Casetext](https://www.thomsonreuters.com/en/press-releases/2023/august/thomson-reuters-completes-acquisition-of-casetext-inc)
- [Meta / Scale AI, $14.3B for 49%](https://www.cnbc.com/2025/06/12/scale-ai-founder-wang-announces-exit-for-meta-part-of-14-billion-deal.html)
- [New Relic / Francisco Partners and TPG](https://www.franciscopartners.com/media/new-relic-to-be-acquired-by-francisco-partners-and-tpg-for-65-billion)

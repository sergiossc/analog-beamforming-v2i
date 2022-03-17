function plot_pattern()
    Nt = 64;
    NtRF = 4;
    Nr = 16;
    NrRF = 4;
    c = 3e8;
    fc = 60e9;
    
    lambda = c/fc;
    txarray = phased.PartitionedArray(...
        'Array',phased.URA([sqrt(Nt) sqrt(Nt)],lambda/2),...
        'SubarraySelection',ones(NtRF,Nt),'SubarraySteering','Custom');
    rxarray = phased.PartitionedArray(...
        'Array',phased.URA([sqrt(Nr) sqrt(Nr)],lambda/2),...
        'SubarraySelection',ones(NrRF,Nr),'SubarraySteering','Custom');
    load('eigen_beam_direction_ch1');
    F = eigen_beam_direction;
    pattern(txarray,fc,-90:90,-90:90,'Type','power','ElementWeights',F,'PropagationSpeed',c, 'Normalize', false);
    saveas(gcf, 'my_beam_direction_pattern_ch1.png');

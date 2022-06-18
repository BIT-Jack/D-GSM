%% Process dataset into mat files %%

clear;
clc;

%% Inputs:
% Locations of raw input files:
MA_000 = 'INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_SR/vehicle_tracks_000.csv';
MA_001 = 'INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_SR/vehicle_tracks_001.csv';
MA_002 = 'INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/DR_USA_Roundabout_SR/vehicle_tracks_002.csv';



%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id||track_id
3: Frame Number||timestamp
4: Local X||x
5: Local Y||y
6: Heading||psi_rad
7: Lateral maneuver
8: Longitudinal maneuver
9-47: Neighbor Car Ids at grid location
%}



%% Load data and add dataset id
disp('Loading data...')
traj{1} = load(MA_000);
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);    %%add Dataset column
traj{2} = load(MA_001);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(MA_002);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
%traj{4} = load(i80_1);    
%traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
%traj{2} = load(MA_001);
%size(traj{2})
%traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
%size(traj{2})
%traj{6} = load(i80_3);
%traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);
numds = 3
for k = 1:numds
    traj{k} = traj{k}(:,[1,2,3,6,7,10]); %%extract: DatasetID, track_id, timestamp,x,y, psi_rad
    size(traj{k})
end

vehTrajs{1} = containers.Map;  %%containers.Map includes   key and key's value,i.e. key-value
vehTrajs{2} = containers.Map;
vehTrajs{3} = containers.Map;
%vehTrajs{4} = containers.Map;
%vehTrajs{2} = containers.Map;
%vehTrajs{6} = containers.Map;

vehTimes{1} = containers.Map;
vehTimes{2} = containers.Map;
vehTimes{3} = containers.Map;
%vehTimes{4} = containers.Map;
%vehTimes{2} = containers.Map;
%vehTimes{6} = containers.Map;

%% Parse fields (listed above):
disp('Parsing fields...')
width_lane = 2;%grid design, width of a grid , length is set to 5meter, 
%2*5/grid
%total : 13x3grid
%total size: len- 5x13=65meter; wid- 3x2=6meter,out_of_dist =65/2=32.5meter
for ii = 1:numds
    vehIds = unique(traj{ii}(:,2));%%extract every vehIds appeared

    for v = 1:length(vehIds)  %%containers.Map: vehTrajs ---> key:vehIds - value:traj info
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:);
    end
    
    timeFrames = unique(traj{ii}(:,3));%%extract every timeFrame captured

    for v = 1:length(timeFrames)%%containers.Map: vehTimes ---> key:FrameIds - value:traj inf
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end
    
    for k = 1:length(traj{ii}(:,1))     %each   
        time = traj{ii}(k,3);   %%FrameID
        dsId = traj{ii}(k,1);   %%DatasetID
        vehId = traj{ii}(k,2);  %%vehID
        vehtraj = vehTrajs{ii}(int2str(vehId));%%the values of vehTrajs i.e.the all info of this Id
        ind = find(vehtraj(:,3)==time);        %%return the position, i.e. the index of TimeFrame
        ind = ind(1);
        center_x = traj{ii}(k,4);
        
        
        
        % Get lateral maneuver:   %%label, add to column 7 [ub-lb]
        ub = min(size(vehtraj,1),ind+40);%upperboundary, the max ub
        lb = max(1, ind-40);             %lowerboundary, the min lb
        %
        if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6)
            traj{ii}(k,7) = 1;
        elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
            traj{ii}(k,7) = 2;
        else
            traj{ii}(k,7) = 3;
        end
        
        
        % Get longitudinal maneuver:  %%label, add to column 8
        ub = min(size(vehtraj,1),ind+50);
        lb = max(1, ind-30);
        if ub==ind || lb ==ind
            traj{ii}(k,8) =1;
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);  %%average longitudinal velocity of history 
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);   %%average longitudinal velocity of future
            if vFut/vHist <0.8
                traj{ii}(k,8) =2;   %%brake
            else
                traj{ii}(k,8) =1;   %%normal
            end
        end

        % Get grid locations:

        t = vehTimes{ii}(int2str(time));%针对于同一时刻 for the same timestamp
        frameEgo = t((center_x-0.5*width_lane)<t(:,4) & t(:,4)<(center_x+width_lane),:);
        frameL = t((center_x-1.5*width_lane)<t(:,4) & t(:,4)<(center_x-0.5*width_lane),:);
        frameR = t((center_x+0.5*width_lane)<t(:,4) & t(:,4)<(center_x+1.5*width_lane) ,:);
       
        if ~isempty(frameL)
            for l= 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5); %5th column is Local Y
                if abs(y) <32.5
                    if y>0
                        if y-2.5<0
                            gridInd=7;
                        else
                            gridInd=8+floor((y-2.5)/5);
                        end
                    else
                        if y>-2.5
                            gridInd=7;
                        else
                            gridInd=6-floor((-y-2.5)/5);
                        end
                    end
                    traj{ii}(k,8+gridInd) = frameL(l,2);
                end
            end
        end
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5);
            if abs(y) <32.5 && y~=0
                    if y>0
                        if y-2.5<0
                            gridInd=20;
                        else
                            gridInd=21+floor((y-2.5)/5);
                        end
                    else
                        if y>-2.5
                            gridInd=20;
                        else
                            gridInd=19-floor((-y-2.5)/5);
                        end
                    end
                    traj{ii}(k,8+gridInd) = frameEgo(l,2);
            end
        end
       if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <32.5
                    if y>0
                        if y-2.5<0
                            gridInd=33;
                        else
                            gridInd=34+floor((y-2.5)/5);
                        end
                    else
                        if y>-2.5
                            gridInd=33;
                        else
                            gridInd=32-floor((-y-2.5)/5);
                        end
                    end
                    traj{ii}(k,8+gridInd) = frameR(l,2);
                end
            end
       end
       %size_check=size(traj{ii})
       %
       %when ZS scenario
       %traj{ii}(k,47)=0;
      
        
    end
end
for jj = 1:numds
    size(traj{jj})
end
%% Split train, validation, test
disp('Splitting into train, validation and test sets...')

trajAll = [traj{1};traj{2};traj{3}];
clear traj;

trajTr = [];

for k = 1:numds
    trajTr = [trajTr;trajAll(trajAll(:,1)==k, :)];%%specific dataset and pick a part of values accroding to the condition
end

 tracksTr = {};
for k = 1:numds
    trajSet = trajTr(trajTr(:,1)==k,:);
    trajset_tr = size(trajSet)
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),[3,4,5,7,8,6])';
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing

disp('Filtering edge cases...')

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    if tracksTr{trajTr(k,1),trajTr(k,2)}(1,15) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>t+1
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);
%% Save mat files:
disp('Saving mat files...')

traj = trajTr;
tracks = tracksTr;
save('scenarios_rowmat/Scenario-MA','traj','tracks');

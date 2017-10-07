function Penalty = KVPenaltyFunction(BoundaryData,UpperName,LowerName,Limit)
%% Define upper & lower data %%
UpperData=cell2mat(BoundaryData(ismember(BoundaryData(:,1),UpperName),2));
LowerData=cell2mat(BoundaryData(ismember(BoundaryData(:,1),LowerName),2));
%% Build penalty function %%
Upper=(UpperData-Limit);
Upper(Upper<=0)=0;
Lower= (Limit-LowerData);
Lower(Lower<=0)=0;
Penalty = mean(Upper) + mean(Lower);
end

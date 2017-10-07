clear all
cd('C:\Users\Jairo F Gudiño R\Desktop\Project2')
FileName='Sovereign Bonds Ratings.xlsx';
FileA='Bond Ratings';
FileB='CDS-5Y';
FileC='Regions';
FileD='Classification Test';
FileClass='Reclassification.xlsx';
FileE='S&P';
%% LECTURA DE INSUMOS %%

windowlength=22;
% Ratings de Bonos %
% Countries={'Brazil','Colombia','Mexico','Peru','Venezuela'};
Agencies={'S&P'};
[~,~,Ratingsraw]=xlsread(FileName,FileA);
Ratingsraw=Ratingsraw(ismember(Ratingsraw(:,1),Agencies'),:);
Ratingsraw=Ratingsraw(:,[2 4 5]);
% Dates %
[Month,Others]=strtok(Ratingsraw(:,2));
Months={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};
for i=1:numel(Months)
Month=strrep(Month,Months(i),num2str(i));
end
[Day,Year]=strtok(Others);
for i=1:length(Day)
DayS=Day{i,1};
if ismember(DayS(1),char('0'))==1
Day{i,1}=str2double(DayS(2));
   else
Day{i,1}=str2double(DayS);
end
end
Month=cellfun(@str2num,Month,'UniformOutput',0);
Year=cellfun(@str2num,Year,'UniformOutput',0);
Dates=datetime(cell2mat(Year),cell2mat(Month),cell2mat(Day));
Ratingsraw(:,2)=cellstr(datestr(cellstr(char(Dates)),'dd/mm/yyyy'));
clear Dates Day Month Year DaysS Others

% Series de Credit Default Swaps para Metodología KV y entrenamiento de algoritmos %
[~,~,Curvesraw]=xlsread(FileName,FileB);
CurveDates=((Curvesraw(2:end,1)));
CurveDatesT=datetime(year(CurveDates,'dd/mm/yyyy'),month(CurveDates,'dd/mm/yyyy'),day(CurveDates,'dd/mm/yyyy'));
CurveDates=cellstr(datestr(cellstr(char(CurveDatesT)),'dd/mm/yyyy'));
% Series de Credit Default Swaps para clasificación %
[~,~,Classraw]=xlsread(FileName,FileD);
Classraw=Classraw(1:230,[1:4 6 7 9]);
CurveCDates=((Classraw(2:end,1)));
CurveCDatesT=datetime(year(CurveCDates,'dd/mm/yyyy'),month(CurveCDates,'dd/mm/yyyy'),day(CurveCDates,'dd/mm/yyyy'));
CurveCDates=cellstr(datestr(cellstr(char(CurveCDatesT)),'dd/mm/yyyy'));
ClassP=find(ismember(CurveDates,CurveCDates(windowlength:end,:)));
%% CONSTRUCCIÓN DE HISTÓRICOS DE CDS 5Y PARA ENTRENAMIENTO %%
CurvesNames=Curvesraw(1,2:end);
Curvesraw=Curvesraw(2:end,2:end);
for i=1:size(Curvesraw,2)
lognan=cellfun(@(v) any(~isnumeric(v(:))),Curvesraw(:,i));
Curvesraw(lognan,i)=num2cell(str2double('NaN'));
end
clear lognan
Curvesraw=num2cell(CompleteValues(cell2mat(Curvesraw),'double'));
%% CONSTRUCCIÓN DE HISTÓRICOS DE CDS 5Y PARA CLASIFICACIÓN %%
Classraw=Classraw(2:end,2:end);
for i=1:size(Classraw,2)
lognan=cellfun(@(v) any(~isnumeric(v(:))),Classraw(:,i));
Classraw(lognan,i)=num2cell(str2double('NaN'));
end
clear lognan
Classraw=num2cell(CompleteValues(cell2mat(Classraw),'double'));
ns=size(Classraw,2);
%% CONSTRUCCIÓN DE HISTÓRICOS DE RATINGS DE BONOS SOBERANOS %%
nc=length(CurvesNames);
RatingsHistory=cell(length(CurveDates),nc);
for i=1:nc
RatingsrawF=Ratingsraw(ismember(Ratingsraw(:,end),CurvesNames(i)),:);
[a,b]=ismember(CurveDates,RatingsrawF(:,2));
DatesTable=sortrows([find(a),b(a)],2);
RatingsHistory(DatesTable(:,1),i+1)=RatingsrawF(DatesTable(:,2),1);
RatingsHistory(1,i+1)=RatingsrawF(size(DatesTable,1)+1,1);
end
clear DatesTable a b RatingsrawF
RatingsHistory=CompleteValues(RatingsHistory(:,2:end),'cell');
[~,~,Ratingsclass]=xlsread(FileClass,FileE);
% Reconversión a categorías predefinidas por Marmi, Nassigh & Regoli (2014)
Ratingorder={'AAA';'AA';'A';'BBB';'BB'};
RnumHistory=cell(length(CurveDates),nc);
for i=1:(nc)
for j=2:size(Ratingsclass,1)
logconv=ismember(RatingsHistory(:,i),char(Ratingsclass(j,1)));
RatingsHistory(logconv,i)=cellstr(repmat(char(Ratingsclass(j,2)),sum(logconv),1));
RnumHistory(logconv,i)=(repmat(Ratingsclass(j,3),sum(logconv),1));
end
end
clear Ratingsclass logconv
%% CREACIÓN DE VENTANAS MÓVILES: APLICACIÓN DE ALG. DE CLASIFICACIÓN Y APROX. KV APPROACH PARA DEF. DE LÍMITES %%
Boundaries=zeros(length(CurveDates)-windowlength,length(Ratingorder)-1);
numbers=1:(length(Ratingorder)-1);

classpos=(windowlength:windowlength:(windowlength*nc))';
KVRatings=cell(length(CurveDates)-windowlength,nc);
KNNRatings=cell(length(CurveDates)-windowlength,nc);
NBRatings=cell(length(CurveDates)-windowlength,nc);
CTRatings=cell(length(CurveDates)-windowlength,nc);

% ClassificationFore=cell(size(Classraw,1)-windowlength,3);
% ClassificationMdl=cell(size(Classraw,1)-windowlength,3);

for i=(windowlength):length(CurveDates)
    
% Definición de matrices de datos %
CurvesrawF=Curvesraw(i-windowlength+1:i,:);
RatingsF=RatingsHistory(i-windowlength+1:i,:);
FinalRaw=cell(windowlength*nc,2);
for c=1:nc
if c==1
FinalRaw(1:windowlength,:)=[RatingsF(:,1),CurvesrawF(:,1)];
  else
FinalRaw(((c-1)*windowlength)+1:c*windowlength,:)=[RatingsF(:,c),CurvesrawF(:,c)];
end
end
clear RatingsF

% Aplicación de Metodologías de Clasificación %
BDTy = (ordinal(FinalRaw(:,1)))';
% 1.  Metodología de K-Nearest Neighbors %
MdlKNN = fitcknn(cell2mat(FinalRaw(:,2)),BDTy,'NumNeighbors',4);
PredKNN = predict(MdlKNN,cell2mat(FinalRaw(:,2)));
KNNRatings(i-windowlength+1,:)=(cellstr(PredKNN(classpos)))';
% 2.  Metodología de Naïve Bayes %
MdlNB = fitcnb(cell2mat(FinalRaw(:,2)),BDTy);
PredNB = predict(MdlNB,cell2mat(FinalRaw(:,2)));
NBRatings(i-windowlength+1,:)=(cellstr(PredNB(classpos)))';
% 3.  Metodología de Classification Trees %
MdlCT = fitctree(cell2mat(FinalRaw(:,2)),BDTy);
PredCT = predict(MdlCT,cell2mat(FinalRaw(:,2)));
CTRatings(i-windowlength+1,:)=(cellstr(PredCT(classpos)))';

% Predicción de ratings out-of-the-sample %

% if sum(ismember(ClassP,i))~=0
% ClassrawF=Classraw(i-ClassP(1)+1:i-ClassP(1)+windowlength,:);
% nFinalRaw=cell(windowlength*ns,1);
% for c=1:ns
% if c==1
% nFinalRaw(1:windowlength)=ClassrawF(:,1);
% else
% nFinalRaw(((c-1)*windowlength)+1:c*windowlength)=ClassrawF(:,c);
% end
% end
% 
% 1.  Predicción por K-Nearest Neighbors %
% ClassificationFore(i-ClassP(1)+1,1) = {predict(MdlKNN,cell2mat(nFinalRaw))};
% 2.  Predicción por Naïve Bayes %
% ClassificationFore(i-ClassP(1)+1,2) = {predict(MdlNB,cell2mat(nFinalRaw))};
% 3.  Predicción por Classification Trees %
% ClassificationFore(i-ClassP(1)+1,3) = {predict(MdlCT,cell2mat(nFinalRaw))};
% 
% end

% Metodología de Kou-Varotto (2008) para definición de límites %
% Creación de tabla de frecuencias de categorías %
RatingFreq=[unique(FinalRaw(:,1)),num2cell(countmember(unique(FinalRaw(:,1)),FinalRaw(:,1)))];
[logic,order]=ismember(Ratingorder,RatingFreq(:,1));
order(order==0)=[];
RatingFreq=[RatingFreq(order,1),RatingFreq(order,2)];
BoundariesN=length(RatingFreq)-1;
% Minimización de la función de penalización
KVBoundaries=zeros(BoundariesN,1);
for h=1:BoundariesN
BoundaryData=FinalRaw(ismember(FinalRaw(:,1),RatingFreq([h h+1]',1)),:);
KV = @(b) KVPenaltyFunction(BoundaryData,RatingFreq(h,1),RatingFreq(h+1,1),b);
initvalue=mean(cell2mat(BoundaryData(:,2)));
options=optimset('Display','off','TolFun',1e-8,'TolX',1e-4);
KVBoundaries(h)=fmincon(KV,initvalue,[],[],[],[],eps,Inf,[],options);
end
clear BoundaryData FinalRaw
% Corrección de límites debido a cruzamientos %
for h=2:length(KVBoundaries)
if (gt(KVBoundaries(h),KVBoundaries(h-1)))==0
KVBoundaries(h)=KVBoundaries(h-1);
end
end
% Almacenamiento secuencial de límites %
logic=(logic(2:end).*logic(1:end-1));
nologic=length(logic)-sum(logic);
Boundaries(i-windowlength+1,numbers(~logic'))=repmat(str2double('NaN'),1,nologic);
logic=numbers'.*logic;
logic(logic==0)=[];
Boundaries(i-windowlength+1,logic')=KVBoundaries';
% Obtención de calificaciones implícitas según definiciones dinámicas de límites%
KVRatingF=cell2mat(Curvesraw(i,:));
for h=1:nc
loglim=logical(KVRatingF(h)>=KVBoundaries);
if sum(loglim)==0
KVRatings(i-windowlength+1,h)=cellstr(Ratingorder{1});
   else
KVRatings(i-windowlength+1,h)=cellstr(Ratingorder{sum(loglim)+1});
end
end

end
% plot(Boundaries)

%% ANÁLISIS DE AJUSTE DE DATOS PARA SELECCIÓN DE ALGORITMO DE CLASIFICACIÓN %%

Alg={'KV','KNN','NB','CT'};
NGap=[-4,-3,-2,-1,0,1,2,3,4];
lNGap=length(NGap);
N={'1','2','3','4','5'};
GapWindow=[30,90,180,360]; %Número de días de comparación%
Ratingorder=Ratingorder';
HistoricalN=double(categorical(RatingsHistory(windowlength:end,:),Ratingorder,N,'Ordinal',true));

for a=1:length(Alg)
eval(['AlgN=double(categorical(',strcat(Alg{a},'Ratings'),',Ratingorder,N));'])
for p=1:length(GapWindow)

RatingGap=zeros(lNGap,lNGap,size(AlgN,1)-GapWindow(p)+1);
confusionMatT=zeros(length(N),length(N),size(AlgN,1)-GapWindow(p)+1);

for t=GapWindow(p):size(AlgN,1)
confusionMatT(:,:,t-GapWindow(p)+1)= confusionmat(AlgN(t-GapWindow(p)+1,:),HistoricalN(t,:));
RatingGapM=AlgN(t,:)-AlgN(t-GapWindow(p)+1,:);
RatingGapO=HistoricalN(t,:)-HistoricalN(t-GapWindow(p)+1,:);
% Matrix to be generated. Rows: Clases KV (-4,-3,..); Columns: Credit Rating Agencies (-4,-3,..).
for q=1:lNGap
logcond=ismember(RatingGapM,NGap(q));
RatingGap(q,:,t-GapWindow(p)+1)=histcounts(RatingGapO(logcond),NGap(1):NGap(end)+1);
end
end

% Cumulative Contingency Table %
RatingContingency=sum(confusionMatT,3);
% Accuracy, Precision, Recall & F1-Scores %
accuracy = sum(diag(RatingContingency))/(nc*(size(AlgN,1)-GapWindow(p)+1));
precision = (diag(RatingContingency)./sum(RatingContingency,2))';
recall = (diag(RatingContingency)./sum(RatingContingency,1)')';
f1Scores = (2*(precision.*recall)./(precision+recall))';
% Gap-Conditioned Contingency Table %
RatingGap=sum(RatingGap,3);
RatingGap=[sum(RatingGap(:,NGap<0),2),sum(RatingGap(:,NGap==0),2),...
    sum(RatingGap(:,NGap>0),2)];
% bar(RatingGap,'stacked')

eval([strcat('Accuracy_',Alg{a},num2str(GapWindow(p))),'=accuracy;'])
eval([strcat('Precision_',Alg{a},num2str(GapWindow(p))),'=precision;'])
eval([strcat('Recall_',Alg{a},num2str(GapWindow(p))),'=recall;'])
eval([strcat('F1Scores_',Alg{a},num2str(GapWindow(p))),'=f1Scores;'])
eval([strcat('CondTable_',Alg{a},num2str(GapWindow(p))),'=RatingContingency;'])
eval([strcat('GapCondTable_',Alg{a},num2str(GapWindow(p))),'=RatingGap;'])

end
end

%% ESTIMACIÓN DE TRANSITION PROBABILITIES POR REGIONES %%
% Construcción recursiva de insumo de estimación %
z=size(CTRatings,1);
TransRaw=cell(z*nc,3);
Dates=cellstr(char(CurveDatesT(windowlength:end)));
for c=1:nc
if c==1
TransRaw(1:z,:)=[repmat(CurvesNames(c),z,1),Dates,CTRatings(:,1)];
  else
TransRaw(((c-1)*z)+1:c*z,:)=[repmat(CurvesNames(c),z,1),Dates,CTRatings(:,c)];
end
end

% Identificación de regiones %
% Regiones definidas por León, Pérez y Leiton (2013)%
[~,~,Regions]=xlsread(FileName,FileC);
Regions=Regions(2:(nc+1),1:2);
LatinAmerica=find(strcmp(Regions(:,2),'L'))';
AsiaOceanAfr=find(strcmp(Regions(:,2),'A'))';
USWEurope=find(strcmp(Regions(:,2),'E'))';
EEurope=find(strcmp(Regions(:,2),'W'))'; % Excluida porque sólo contiene a Rusia %

% Estimación por métodos:

startDate = Dates(1);
endDate = Dates(end);

%(A) Duración
% [~,~,idTotalsD] = transprob(TransRaw,'startDate',startDate,'endDate',endDate,...
%     'labels',Ratingorder);
% 
% transMatLatinAmericaD1Y = transprobbytotals(idTotalsD(LatinAmerica));
% transMatAsiaOceanAfrD1Y = transprobbytotals(idTotalsD(AsiaOceanAfr));
% transMatUSWEuropeD1Y = transprobbytotals(idTotalsD(USWEurope));

% (B) Cohort

% By default, the cohort algorithm internally gets yearly snapshots of the credit ratings, but the number of snapshots per year is definable using the parameter/value pair snapsPerYear. 
% To get the estimates using quarterly snapshots:
% transMat3 = transprob(data,'startDate',startDate,'endDate',endDate,...
% 'algorithm','cohort','snapsPerYear',4)
% Estimate Point-in-Time and Through-the-Cycle Probabilities %

TimeIntervalName={'1Y','2Y','3Y','4Y','5Y'};
TimeIntervalNumber=[360,720,1080,1440,1800]./360;
LogT='transInterval';

[~,~,idTotalsC] = transprob(TransRaw,'startDate',startDate,'endDate',endDate,...
    'algorithm','cohort','labels',Ratingorder);
for t=1:length(TimeIntervalNumber)
eval([strcat('transMatLatinAmericaC',TimeIntervalName{t}),'=transprobbytotals(idTotalsC(LatinAmerica),LogT,TimeIntervalNumber(t));'])
eval([strcat('transMatAsiaOceanAfrC',TimeIntervalName{t}),'=transprobbytotals(idTotalsC(LatinAmerica),LogT,TimeIntervalNumber(t));'])
eval([strcat('transMatUSWEuropeC',TimeIntervalName{t}),'=transprobbytotals(idTotalsC(LatinAmerica),LogT,TimeIntervalNumber(t));'])
end
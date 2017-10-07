function MatrixP = CompleteValues(Matrix,type)
% Creado por: Jairo F. Gudiño R.
% Missing values are completed with available information, repeating
% first values with the first observed value, and the others
% with the last observed price.

% Matrix must be double.
MatrixP=Matrix;
for z=1:size(Matrix,2)
SecPrice=Matrix(:,z);
if strcmp(type,'double')
Pos=find(~isnan(SecPrice));
elseif strcmp(type,'cell')
logPrice=logical(cell2mat(cellfun(@(x) (ischar(x)),SecPrice,'UniformOutput',0)));
Pos=find(logPrice);
end
if size(Pos,1)~=0
SecPos=SecPrice(Pos);
if size(SecPos,1)~=0
for t=1:size(SecPos,1)
if size(SecPos,1)==1
MatrixP(:,z)=(repmat(SecPos,size(SecPrice,1),1)); %num2cell
  else
if t~=size(SecPos,1)
if t==1
MatrixP(1:Pos(t+1)-1,z)=(repmat(SecPos(t),size(1:Pos(t+1)-1,2),1));
   else
MatrixP(Pos(t):Pos(t+1)-1,z)=(repmat(SecPos(t),size(Pos(t):Pos(t+1)-1,2),1));
end
   else
MatrixP(Pos(t):end,z)=(repmat(SecPos(t),size(Pos(t):size(SecPrice,1),2),1));
end
end
end
end
end
end
end
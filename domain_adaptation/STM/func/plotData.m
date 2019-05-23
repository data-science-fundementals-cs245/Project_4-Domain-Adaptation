function plotData(X,Y,Xte,Yte)

% plot training sample
for i = 1:size(X,1)
  if Y(i) == 1
    pat='cs'; 
  else
    pat='co'; 
  end
  plot(X(i,1),X(i,2),pat,'markerfacecolor',[.65,.65,.65],...
    'markeredgecolor',[.2,.2,.2]);
end

% plot test sample
for i = 1:size(Xte,1)
  if Yte(i) == 1
    pat='rs';
  else
    pat='ro';
  end
  plot(Xte(i,1),Xte(i,2),pat,'markerfacecolor',[.98,.98,.98],...
    'markeredgecolor',[.6,0,.6]);
end
function plotSepBg(X,model)

w = model.w;
b = model.b;

% get boundary points
x = [min(X(:,1)),max(X(:,1))];
y = [min(X(:,2)),max(X(:,2))];
axis([x,y]);
[xx,yy] = meshgrid(x(1):.005:x(2), y(1):.005:y(2));
XX = [xx(:),yy(:)];

% get labels for boundary points
label = sign(XX*w+b);
posind = label == 1;
negind = label ~= 1;

% plot decision hyperplane
plot(xx(posind),yy(posind),'bs','markersize',1.5,'MarkerFaceColor',[.70,.75,1],'MarkerEdgeColor','none');
plot(xx(negind),yy(negind),'rs','markersize',1.5,'MarkerFaceColor',[1,.85,.80],'MarkerEdgeColor','none');
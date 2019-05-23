function plotResults(X,Y,Xte,Yte,idealModel,genModel,stmModel,iIter,log)
% plot results

hf=figure(1); clf; hold on; set(gca,'xtick',[],'ytick',[]); box on;
pp=get(gcf,'position'); set(gcf,'position',[pp(1:2),320,320]),

% plot stm seperation
plotSepBg(X,stmModel); 

% plot data
plotData(X,Y,Xte,Yte); 

% plot hyperplane
hIdeal = drawPlane(idealModel.w, -idealModel.b, [.9,.1,.1],'--');
hGen   = drawPlane(genModel.w, -genModel.b, [1,.6,0],'-.');
hStm   = drawPlane(stmModel.w, -stmModel.b, [.6,0,0]);

% compute STM training/testing accuracy
accTe = sum(sign(Xte*stmModel.w + stmModel.b)==Yte)/length(Yte)*100;
accTr = sum(sign(X*stmModel.w + stmModel.b)==Y)/length(Y)*100;

% put on info
legend([hGen,hStm,hIdeal], {'Generic','STM','Ideal'});
title(sprintf('it#%d (Tr%%=%.1f, Te%%=%.1f)',iIter,accTr,accTe));

% plot objective
plotObj(log,iIter,hf);


function plotObj(log,iIter,hf)
% plot objective curve

figure(2); pp = get(hf,'position'); 
set(gcf,'position',[pp(1)+320,pp(2),480,320]);
plot(log.obj(1:iIter)); grid on;
title('Convergence curve');
xlabel('#Iteration'); ylabel('Objective');
xx = get(gca,'xlim'); 
set(gca,'xlim',[1,xx(2)],'xtick',1:iIter);
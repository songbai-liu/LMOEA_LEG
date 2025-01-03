classdef LMOEAD < ALGORITHM
% <multi> <real/integer>
% Learnable multiobjective evolutionary algorithm

    methods
        function main(Algorithm,Problem)
            %% Parameter settings
            k = 2;
            Population     = Problem.Initialization();
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            %Archive       = [Population,Problem.Initialization()];
            Lower = Problem.lower;
	        Upper = Problem.upper;

            %% Optimization
            while Algorithm.NotTerminated(Population) 
                 % Prepareing training data and learn the MLP model
                [INDEX,DIS] = Association(Population,V,k);
                for i = 1:Problem.N
                    if DIS(INDEX(1,i),i) < DIS(INDEX(2,i),i)
                        winner(i) = Population(INDEX(1,i));
                        losser(i) = Population(INDEX(2,i));            
                    else
                        winner(i) = Population(INDEX(2,i));            
                        losser(i) = Population(INDEX(1,i));            
                    end    
                end 
                mlp = ModelLearning(Problem, losser, winner); 

                N = length(Population);
                zmin = min(Population.objs,[],1);
                zmax = max(Population.objs,[],1);
                PopObj    = (Population.objs - repmat(zmin,N,1))./(repmat(zmax-zmin,N,1));
                B = pdist2(PopObj,PopObj);
                [~,B] = sort(B,2);
                B = B(:,1:10);
                %Reproduction
                for i = 1 : Problem.N
                    t = Problem.FE/Problem.maxFE;
                    child = Population(i).decs;
                    [N, D] = size(child);
                    Site = rand(N,D) < 0.75;
                    p = B(i,randperm(end));
                    P = randperm(Problem.N);
                    if rand < 0.8
                        Parent1 = Population(p(1)).decs;
		                Parent2 = Population(p(2)).decs;
                    else
                        Parent1 = Population(P(1)).decs;
		                Parent2 = Population(P(2)).decs;
                    end
                    if rand > t  %search in the improvement representation space
                        child = (child-Lower)./(Upper-Lower);
                        Parent1 = (Parent1-Lower)./(Upper-Lower);
                        Parent2 = (Parent2-Lower)./(Upper-Lower);
						[GDV,~] = mlp.forward(child);
                        [GDV1,~] = mlp.forward(Parent1);
                        [GDV2,~] = mlp.forward(Parent2);
                        for j = 1 : 2
                            [GDV,~] = mlp.forward(GDV);
                            [GDV1,~] = mlp.forward(GDV1);
                            [GDV2,~] = mlp.forward(GDV2);
                        end
						child = GDV.*repmat(Upper-Lower,size(GDV,1),1) + repmat(Lower,size(GDV,1),1);
                        RGDV1 = GDV1.*repmat(Upper-Lower,size(GDV1,1),1) + repmat(Lower,size(GDV1,1),1);
                        RGDV2 = GDV2.*repmat(Upper-Lower,size(GDV2,1),1) + repmat(Lower,size(GDV2,1),1);
                        child(Site) = child(Site) + 0.5*(RGDV1(Site)-RGDV2(Site));
                    else %search in the original space
                        child(Site) = child(Site) + 0.5*(Parent1(Site)-Parent2(Site));
                    end
                    %%mutation
                    child = RealMutation(child,Lower,Upper);
                    %Evaluation of the new child
		            child = Problem.Evaluation(child);
		            %add the new child to the offspring population
		            Offspring(i) = child;  
                end
                
                %Environmental selection
                Population = EnvironmentalSelection_Clustering([Population,Offspring],Problem.N,Problem.N);
            end
        end
    end
end

function [INDEX,DIS] = Association(Population,V,k)
    % Normalization 
    N = length(Population);
    zmin = min(Population.objs,[],1);
    zmax = max(Population.objs,[],1);
    PopObj    = (Population.objs - repmat(zmin,N,1))./(repmat(zmax-zmin,N,1));
    % Associate k candidate solutions to each reference vector
    normP  = sqrt(sum(PopObj.^2,2));
    Cosine = 1 - pdist2(PopObj,V,'cosine');
    d1     = repmat(normP,1,size(V,1)).*Cosine;
    d2     = repmat(normP,1,size(V,1)).*sqrt(1-Cosine.^2);
    DIS    = d1 + 0.0*d2;
    [~,index] = sort(d2,1);
    INDEX     = index(1:min(k,length(index)),:);
end

function Offspring = RealCrossover(Parent1,Parent2)
% Simulated binary crossover
    disC  = 20;
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
                 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
end

function Offspring = RealMutation(Offspring,Lower,Upper)
% Polynomial mutation
    disM  = 20;
    [N,D] = size(Offspring);
    Lower = repmat(Lower,N,1);
    Upper = repmat(Upper,N,1);
    Site  = rand(N,D) < 1/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring       = min(max(Offspring,Lower),Upper);
end

function mlp = updateMode(mlp, Archive, V, k, Problem)
    [INDEX,DIS] = Association(Archive,V,k);
    for i = 1:Problem.N
        if DIS(INDEX(1,i),i) < DIS(INDEX(2,i),i)
           winner(i) = Archive(INDEX(1,i));
           losser(i) = Archive(INDEX(2,i));            
        else
           winner(i) = Archive(INDEX(2,i));            
           losser(i) = Archive(INDEX(1,i));            
        end    
     end 
     mlp = ModelLearning(Problem, losser, winner);
end

function Offspring = SBX(Parent1,Parent2)
% Genetic operators for real and integer variables
    [proC,disC] = deal(1,20);
    %% Simulated binary crossover
    [N,D] = size(Parent1);
    beta  = zeros(N,D);
    mu    = rand(N,D);
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],N,D);
    beta(rand(N,D)<0.5) = 1;
    beta(repmat(rand(N,1)>proC,1,D)) = 1;
    Offspring = (Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2;
end
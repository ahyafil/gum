classdef sparsearray
    %SPARSEARRAY Class-of N-day array with sparse coding
    % sparsearray(X)
    % sparsearray('empty',[m1 m2 ...mD]) dimension of the array
    % sparsearray('empty',[m1 m2 ...mD],sz) to preallocate
    % sparsearray([I1 I2... In],V) where Ii is column vector of
    % indices
    % sparsearray([I1 I2... In],V, [Imax1 Imax2... Imaxn])
    % See also sparse, issparse


    %!! change subindex to onehotencoding
    properties
        siz
        value
        sub = {}
    end

    methods
        %% CONSTRUCTOR
        function obj = sparsearray(varargin)

            if isa(varargin{1},'sparsearray')
                obj = varargin{1};
            elseif ischar(varargin{1}) && strcmp(varargin{1}, 'empty')
                obj.siz = varargin{2};
                m = prod(obj.siz);
                n =1; % prod(obj.siz(2:end));
                obj.value = sparse([],[],[],m,n,varargin{3:end});

            elseif ischar(varargin{1}) && any(strcmp(varargin{1}, {'rand','randn'}))
                obj.siz = varargin{2};
                m = prod(obj.siz);
                n =1;
                density = varargin{3};

                if strcmp(varargin{1}, 'rand')
                    obj.value = sprand(m,n,density);
                else
                    obj.value = sprandn(m,n,density);
                end


            elseif nargin>=2
                % sparsearray([I1 I2... In],V)
                Sub = varargin{1};
                V = varargin{2};
                [n,D] = size(Sub); % number of non-zero values x dimensions of array

                if any(Sub(:)<=0)
                    error('indices must be positive integers');
                end
                if nargin>3
                    error('too many inputs');
                elseif nargin==3
                    S = varargin{3};
                    if length(S)~=D
                        error('vector of array size must have as many elements as columns in matrix of indices');
                    end
                    for d=1:D
                        if any(Sub(:,d)>S(d))
                            error('subindex in column %d cannot be larger than associate size %d', d, S(d));
                        end
                    end

                else
                    S = max(V,[],1); % maximum index
                end

                Sub = num2cell(Sub,1); % convert to cell array, one cell per dimension
                Ind = sub2ind(S, Sub{:}); % convert to index

                obj.value = sparse(Ind,ones(1,n),V, prod(S),1);
                obj.siz = S;

            else
                %sparsearray(X)
                if issparse(varargin{1})
                    obj.value = varargin{1}(:);
                elseif nnz(varargin{1})<.1*numel(varargin{1}) % make value sparse if needed
                    [I,J,V] = find(varargin{1}(:)); % find indices and values
                    obj.value = sparse(I,J,V, numel(varargin{1}),1);
                else
                    obj.value = varargin{1}(:);
                end
                obj.siz = size(varargin{1});
            end
        end

        %% DISP
        function disp(obj)
            if any(onehotencoding(obj))
                obj = fullcoding(obj);
            end
            if ~issparse(obj.value) || ismatrix(obj)
                disp(obj.value);
            elseif all(obj.value(:) ==0)
                sz = num2cell(obj.siz);
                D = ndims(obj);
                str = ['     All zero sparse: ' repmat('%dx',1,D-1) '%d\n\n']; % string pattern
                fprintf(str, sz{:});
            else
                % n =  numel(obj);
                sz = obj.siz;
                D = ndims(obj);
                S = cell(1,D);
                str = ['(' repmat('%d,',1,D-1) '%d): %f\n']; % string pattern

                V = allvalues(obj);

                for i=find(V)'
                    [S{:}] = ind2sub(sz,i);
                    % fprintf(str, S{:}, full(obj(i)));
                    fprintf(str, S{:}, full(V(i)));
                end

            end
        end

        %% CONCATENATE
        function obj = cat(d, varargin)
            % X = cat(d, X1, X2, ...) concatenates arrays X1, X2,... across dimension
            % d.
            %
            % Note: X does not use one-hot-encoding even if input arrays do.
            varargin = cellfun(@sparsearray, varargin, 'UniformOutput',false); % convert all input to sparse array
            nObj = length(varargin);
            nD = max(cellfun(@ndims, varargin)); % number of dimensions for all array

            S = zeros(1,nD); % size of new array

            for e=setdiff(1:nD,d)
                this_siz = cellfun(@(x) size(x,e), varargin); % number of elements along this dimension
                assert(all(this_siz==this_siz(1)), 'number of elements do not match along non-concatenating dimensions');
                S(e) = this_siz(1);
            end

            this_siz = cellfun(@(x) size(x,d), varargin); % number of elements all concatenating dimension
            S(d) = sum(this_siz);
            shft = cumsum([0 this_siz]); % how much we need to shift position of elements along dimension d in new array

            % get indices and values for all arrays
            I = cell(1,nObj);
            V = cell(1,nObj);
            for o=1:nObj
                [I{o},V{o}] = find(varargin{o});
                I{o}(:,end+1:nD) = 1; % pad with ones if smaller dimension array
                I{o}(:,d) = I{o}(:,d) + shft(o); % shift position along dimension d
            end

            % concatenate indices and values along arrays
            I = cat(1,I{:});
            V = cat(1,V{:});

            % create sparse array
            obj = sparsearray(I,V, S);


        end

        %% SIZE
        function S = size(obj,d)
            S = obj.siz;
            if nargin>1
                if length(S)<d
                    S = 1;
                else
                    S = S(d);
                end
            end
        end

        %% NDIMS
        function n = ndims(obj)
            S = size(obj);
            if any(S==1) % remove singleton dimensions in the end
                lastnonsing = find(S>1, 1,'last');
                if isempty(lastnonsing)
                    lastnonsing = 2;
                end
                S(lastnonsing+1:end) = [];
            end
            n = max(2,length(S));
        end

        %% NUMBER OF ELEMENTS
        function n = numel(obj)
            n = prod(size(obj));
        end

        %% RESHAPE
        function obj = reshape(obj,varargin)
            S = [varargin{:}];

            % reshape sparse tensor
            if prod(S)~=prod(obj.siz)
                error('To RESHAPE the number of elements must not change.');
            end
            if isequal(S,obj.siz)
                return;
            end
            obj = fullcoding(obj);

            obj.siz = S;
            % if ~issparse(obj.value) || length(S)<=2
            %     obj.value = reshape(obj.value,S);
            % end

        end


        %% PERMUTE
        function obj = permute(obj,P)

            D = length(P);
            for d=1:D
                if sum(P==d)~=1
                    error('P should be a permutation');
                end
            end

            % permute value array
            if issparse(obj.value)

                NonOneHot = find(~onehotencoding(obj));
                sz = size(obj);
                Pnon = P(ismember(P, NonOneHot));
                new_sz = sz(Pnon); % size of non-one-hot-encoding array

                sz = sz(NonOneHot);

                nElem = prod(sz); % total number of element of sub-array
                D = length(NonOneHot); % dimensionality

                ind = ones(1,nElem);
                for d=1:D
                    sb = (0:sz(d)-1)*prod(sz(1:d-1)); % sub-indices for this dimension
                    dd = find(Pnon==NonOneHot(d)); % which position in new dimension order
                    if dd>1
                        sb = repelem(sb, prod(new_sz(1:dd-1))); % replicate each value in this dimension according to lower dimensions
                    end
                    if dd<D
                        sb = repmat(sb,1,prod(new_sz(dd+1:end))); % replicate vector of subindices according to higher dimensions
                    end
                    ind = ind + sb;
                end

                obj.value = obj.value(ind);
                obj.value = reshape(obj.value,prod(new_sz),1);

            else
                obj.value = permute(obj.value,P);
            end

            % permute one-hot encoding arrays
            for d=find(onehotencoding(obj))
                obj.sub{d} = permute(obj.sub{d},P);
            end

            % permute order of  one-hot encoding arrays
            obj.sub(end+1:length(P)) = {[]};
            obj.sub = obj.sub(P);

            % update size
            S = size(obj);
            S(end+1:length(P)) = 1;
            obj.siz = S(P);
        end

        %% ISMATRIX
        function bool = ismatrix(obj)
            bool = ndims(obj)<=2;
        end

        %% ISCOLUMN
        function bool = iscolumn(obj)
            S = size(obj);
            bool = all(S(2:end)==1);
        end

        %% ISORW
        function bool = isrow(obj)
            S = size(obj);
            bool = S(1)==1 && all(S(3:end)==1);
        end

        %% CONVERT TO SPARSE MATRIX
        function M = matrix(obj)

            if ~ismatrix(obj)
                error('sparse array has dimension higher than 2');
            end

            M = reshape(allvalues(obj),size(obj));
        end

        %% CONVERT TO DOUBLE
        function obj = double(obj)

            obj.value = double(obj.value);
            %   if ismatrix(obj)
            %       M = matrix(obj);
            %   else
            %       M = full(obj);
            %   end
        end

        %% NUMBER OF NON-ZERO VALUES
        function n = nnz(obj)
            n = nnz(obj.value);
        end

        %% REPLACE NON-ZERO VALUES BY ONES
        function obj = spones(obj)
            obj.value = spones(obj.value);
        end


        %% FULL
        function X = full(obj)
            X = full(allvalues(obj));
            X = reshape(X,obj.siz);
        end

        %% FIND
        function [S, V] = find(obj)
            if any(onehotencoding(obj))
                obj = fullcoding(obj);
            end

            sz = obj.siz;
            D = ndims(obj);

            [I, ~, V] = find(obj.value(:));

            % convert to subscripts
            S = cell(1,D);
            [S{:}] = ind2sub(sz,I);
            S = cat(2,S{:});

        end

        %% ONE-HOT ENCODING (one boolean per dimension)
        function bool = onehotencoding(obj,d)
            bool = ~cellfun(@isempty, obj.sub);
            bool(end+1:ndims(obj)) = false;
            if nargin==1
                d = 1:ndims(obj);
            end
            bool = bool(d);
        end

        %% ISSPARSE
        function bool = issparse(obj)
            bool = true;
        end

        %% ISSPARSEVALUE (whether field value is sparse)
        function bool = issparsevalue(obj)
            bool = issparse(obj.value);
        end

        %% FULLCODING: change one-hot encoding coding to coding
        function obj = fullcoding(obj,d)
            if nargin<2 % if dimension is not provided: for all subindex coding dimensions
                d = find(onehotencoding(obj));
            end
            if isempty(d)
                return;
            end
            if any(~ismember(d, find(onehotencoding(obj))))
                error('not a subindex coding dimension');
            end

            obj.value = sparse(obj.value);

            [Ind,~,V] = find(obj.value(:)); % indices and values of non-zero values

            NonOneHot = setdiff(1:ndims(obj), find(onehotencoding(obj))); % dimensions not coded with one-hot-encoding
            sz = size(obj);
            sz  = sz(NonOneHot);

            if isempty(NonOneHot)
                Sub = {};
            elseif isscalar(NonOneHot)
                Sub = {Ind};
            else
                Sub = cell(1,length(NonOneHot));
                [Sub{:}] = ind2sub(sz, Ind); % indices of values for non-one-hot coding position
            end
            NOHE = NonOneHot;

            for dd=d % for all dimensions to change from one-hot encoding
                SubSub = Sub;
                for f = 1:length(NOHE)
                    if size(obj.sub{dd},NOHE(f)) ==1 % if subindex matrix is singleton along this dim
                        SubSub{f} = ones(size(SubSub{f}));
                    end
                end
                OneHotIndex = sub2ind(size(obj.sub{dd}), SubSub{:}); % indices with one-hot-encoding array
                NewOneHot = obj.sub{dd}(OneHotIndex); % new subindex for this dimension

                rmvdata = NewOneHot ==0;
                if any(rmvdata)
                    for ddd = 1:length(Sub)
                        Sub{ddd}(rmvdata) = [];
                    end
                    V(rmvdata) = [];
                    NewOneHot(rmvdata) = [];
                end


                Sub{end+1} = NewOneHot;
                sz(end+1) = size(obj,dd);

                obj.sub{dd} = [];
                NOHE(end+1) = dd;
            end

            [NonOneHot,ord] = sort([NonOneHot d]);

            % sort dimensions
            Sub = Sub(ord);
            sz = sz(ord);

            Ind = sub2ind(sz, Sub{:});

            obj.value = sparse(Ind, ones(1,length(V)),V, prod(sz),1);
        end

        %% REPMAT
        function obj = repmat(obj,varargin)
            rep = [varargin{:}];

            if all(rep==1) % if no need for replication
                return;
            end
            assert(all(mod(rep,1)==0), 'Replication factors must be a row vector of integers or integer scalars.');

            rep = max(rep,0); % negative values are taken as equal to 0

            if length(rep)<length(obj.siz)
                rep = [rep ones(1, length(obj.siz)-length(rep))];
            end

            if any(rep==0) % output an empty sparse array
                NewSize = obj.siz .* rep;
                obj =  sparsearray('empty',NewSize);
                return;
            end

            % cannot replicate one-hot encoding dimensions
            OneHotDims = find(onehotencoding(obj));
            obj = fullcoding(obj, OneHotDims(rep(OneHotDims)>1));

            if issparse(obj.value) && (~ismatrix(obj) || length(rep)>2)

                NonOneHot = ~onehotencoding(obj);
                sz = size(obj);
                if length(rep)>ndims(obj)
                    NonOneHot = [NonOneHot true(1,length(rep)-ndims(obj))];
                    sz( ndims(obj)+1:length(rep)) = 1;
                end
                RepNonOneHot = rep(NonOneHot);
                sz = sz(NonOneHot);
                new_sz = sz .* RepNonOneHot; % size of new array

                nElem = prod(new_sz); % total number of element of sub-array
                D = sum(NonOneHot); % dimensionality

                % build the vector of indices for new matrix (this is not
                % optimal as this vector is full - should work a more
                % efficient way, perhaps converting to sparse matrix ...)
                ind = ones(1,nElem);
                for d=1:D
                    sb = (0:sz(d)-1)*prod(sz(1:d-1)); % sub-indices for this dimension
                    if RepNonOneHot(d)>1 % if needs to replicate along that dimension
                        sb = repmat( sb, 1, RepNonOneHot(d));
                    end
                    if d>1
                        sb = repelem(sb, prod(new_sz(1:d-1))); % replicate each value in this dimension according to lower dimensions
                    end
                    if d<D
                        sb = repmat(sb,1,prod(new_sz(d+1:end))); % replicate vector of subindices according to higher dimensions
                    end
                    ind = ind + sb;
                end

                obj.value = obj.value(ind);
                obj.value = reshape(obj.value,prod(new_sz),1);

            else
                sz = obj.siz;
                sz(onehotencoding(obj)) = 1;
                obj.value = reshape(obj.value, sz);
                obj.value = repmat(obj.value,varargin{:});
                obj.value = obj.value(:);
            end

            for d=find(onehotencoding(obj))

                % only replicate non-singleton dimensions
                S = size(obj.sub{d});

                R = rep;
                R(length(S)+1:end) = 1;
                R(S==1) = 1;
                obj.sub{d} = repmat(obj.sub{d},R);
            end

            obj.siz(end+1:length(rep)) = 1;
            obj.siz = obj.siz .* rep;

        end

        %% REPEAT ELEMENTS
        function obj = repelem(obj,varargin)
            D = ndims(obj);
            if length(varargin)<D
                varargin(end+1:D) = repmat({1}, 1, D-length(varargin));
            end

            C = cell(1,length(varargin));
            for d=1:length(varargin)
                v = varargin{d};
                s = size(obj,d);
                if ~isequal(v,1)  % if 1, no change along this direction, don't do anything
                    if isscalar(v)
                        v = repmat(v,1,s);
                    end

                    % sparse projection matrix
                    nRow = sum(v);
                    col_idx = repelem(1:s, v);
                    C{d}  = sparse(1:nRow,col_idx,ones(1,nRow));

                end

            end
            obj = tensorprod(obj, C);

        end

        %% ADD ONE-HOT-ENCODING ALONG ONE DIMENSION
        function obj = add_onehotencoding(obj,d, S, Smax)
            % X = add_onehotencoding(X,d, S) to add one-hot encoding to
            % sparse array X with values S along dimension d.
            %
            % X = add_onehotencoding(X,d, S, Smax) to define the maximum
            % value of S, i.e. the size of X along dimension d.

            if nargin<4
                Smax =  max(S(:));
            end

            for dd=setdiff(1:ndims(obj),d)
                if size(S,dd)~=1 && size(S,dd)~=obj.siz(dd)
                    error('incorrect size for subindex array');
                end
            end

            % make sure we use most compact coding of indices
            if Smax<256
                S = uint8(S);
            elseif Smax<65536
                S = uint16(S);
            else
                S = uint32(S);
            end

            % add to object subindices coding
            obj.sub{d} = S;

            % update size of array
            if d>ndims(obj)+1 % add singleton dimensions if required
                obj.siz(ndims(obj)+1:d-1) = 1;
            end
            obj.siz(d) = Smax;

        end

        % we add this to solve issues with nargout ( see
        % https://itectec.com/matlab/matlab-bug-in-subsref-overloading/)
        function n = numArgumentsFromSubscript(obj, s, ic)
            n = builtin('numArgumentsFromSubscript', obj, s, ic);
        end

        %% CALL TO obj.PropertyName or obj(...)
        function varargout = subsref(obj,s)
            switch s(1).type
                case '.'
                    %   if length(s) == 1
                    %       % Implement obj.PropertyName
                    %       ...
                    %   elseif length(s) == 2 && strcmp(s(2).type,'()')
                    %   % Implement obj.PropertyName(indices)
                    %   ...
                    %   else

                    varargout  = cell(1,max(nargout,1));

                    [varargout{:}] = builtin('subsref',obj,s);
                    %   end
                case '()'
                    %                     if length(s) == 1
                    %                         % Implement obj(indices)
                    %                         ...
                    %                     elseif length(s) == 2 && strcmp(s(2).type,'.')
                    %                     % Implement obj(ind).PropertyName
                    %                     ...
                    %                     elseif length(s) == 3 && strcmp(s(2).type,'.') && strcmp(s(3).type,'()')
                    %                     % Implement obj(indices).PropertyName(indices)
                    %                     else
                    %                         % Use built-in for any other expression
                    %                         [varargout{1:nargout}] = builtin('subsref',obj,s);
                    %                     end


                    s1 = s(1);

                    if (~issparse(obj.value)&&~any(obj.onehotencoding)) || isscalar(s1.subs) % syntax: obj(ind1:ind3) or obj(:)
                        if any(onehotencoding(obj))
                            obj= fullcoding(obj);
                        end

                        if ~isscalar(s1.subs)
                            obj.value = reshape(full(obj.value), size(obj));
                        end

                        % return column vector
                        b = subsref(obj.value,s1); %obj.value(s.subs{:});
                    else % e.g. obj(:,2:4,:)

                        D = ndims(obj);
                        AllIndices = strcmp(s1.subs,':'); % boolean vector, true if all indices are taken in corresponding dim

                        % interpret X(v,:) as X(v,:,:,...)
                        if length(s1.subs)==2 && AllIndices(2)
                            s1.subs(3:D) = repmat(s1.subs(2),1,D-2);
                            AllIndices(3:D) = true;
                        end

                        for d = find(~AllIndices & obj.onehotencoding)
                            % do not maintain one-hot encoding if referencing a subset
                            % (technically we could if there's no index repetition)
                            obj = fullcoding(obj, d);
                        end

                        if length(s1.subs)>= D && all(AllIndices(2:D))
                            %syntax: obj(v,:,:,:), only the first dim is indexed
                            S = obj.siz;

                            % get correspinding values for one-hot encoding dimensions
                            for d=find(obj.onehotencoding)
                                if size(obj.sub{d},1)>1
                                    obj.sub{d} = subsref(obj.sub{d}, s1);
                                end
                            end

                            % now deal with value array
                            S(obj.onehotencoding) = 1;
                            obj.value = reshape(obj.value,S(1),prod(S(2:D))); % reshape as matrix grouping all dimensions except first as columns

                            % select corresponding rows
                            s1.subs(3:D) = [];
                            obj.value = subsref(obj.value,s1);

                            % b = subsref(b, s1);
                            obj.siz(1) = size(obj.value,1); %  update size along first dim

                            % size of new array
                            % nsub = S;
                            % nsub(1) = size(b,1); % number of columns in b
                            b = obj;
                        elseif length(s1.subs)>= D && all(AllIndices(1:D-1))
                            %syntax: obj(:,:,:,v), only the last dim is indexed
                            S = obj.siz;

                            % get correspinding values for one-hot encoding dimensions
                            for d=find(obj.onehotencoding)
                                if size(obj.sub{d},D)>1
                                    obj.sub{d} = subsref(obj.sub{d}, s1);
                                end
                            end

                            % now deal with value array
                            S(obj.onehotencoding) = 1;
                            obj.value = reshape(obj.value,prod(S(1:D-1)),S(D)); % reshape as matrix grouping all dimensions except last as rows

                            % select corresponding columns
                            s1.subs(1:D-2) = [];
                            obj.value = subsref(obj.value, s1);

                            obj.siz(D) = size(obj.value,2); %  update size along first dim

                            b = obj;
                            % size of new array
                            % nsub = S;
                            % nsub(D) = size(b,2); % number of columns in b
                        else

                            % get linear indices from sub-indices (this is very
                            % slow if indexing large number of indices
                            % irrespective of sparsity, should be improved!)

                            [ind,nsub] = get_indices(obj.siz, s1.subs);

                            %!!! correct bug here: will not work if there
                            %are still one-hot-encoding dimensions!

                            % extract value
                            b = obj.value(ind);

                            if length(nsub)>1
                                b = sparsearray(b);
                            end
                            b = reshape(b,nsub);

                            if length(nsub)>1
                                b = sparsearray(b);
                            end

                            b = reshape(b,nsub);

                        end

                        %                         if length(nsub)>1
                        %                             b = sparsearray(b);
                        %                         end
                        %  b = reshape(b,nsub);

                    end

                    if length(s)>1 % i.e. for obj(indices).PropertyName(indices)
                        b = subsref(b, s(2:end)); % recursive call
                    end

                    varargout = {b};
                case '{}'
                    error('Brace indexing is not supported for variables of this type.');
                    %                     if length(s) == 1
                    %                         % Implement obj{indices}
                    %                         ...
                    %                     elseif length(s) == 2 && strcmp(s(2).type,'.')
                    %                     % Implement obj{indices}.PropertyName
                    %                     ...
                    %                     else
                    %                     % Use built-in for any other expression
                    %                     [varargout{1:nargout}] = builtin('subsref',obj,s);
                    %                     end
                otherwise
                    error('Not a valid indexing expression')
            end
        end


        %% ASSIGN TO SUB-INDICES
        function obj = subsasgn(obj,s,varargin)

            % Allow subscripted assignment to uninitialized variable
            if isequal(obj,[])
                % obj = ClassName.empty;
            end

            switch s(1).type
                case '.'
                    %    if length(s) == 1
                    %       % Implement obj.PropertyName = varargin{:};
                    %       ...
                    %    elseif length(s) == 2 && strcmp(s(2).type,'()')
                    %       % Implement obj.PropertyName(indices) = varargin{:};
                    %       ...
                    %    else
                    % Call built-in for any other case
                    obj = builtin('subsasgn',obj,s,varargin{:});
                    %    end
                case '()'
                    if length(s)>1
                        error('incorrect assignment for this type');
                    end
                    obj = fullcoding(obj);

                    if ~issparse(obj.value) || length(s.subs)==1 % obj(ind1:ind3) or obj(:), return column vector
                        if isa(s.subs{1}, 'sparsearray') % if nD sparse boolean array
                            s.subs{1} = allvalues(s.subs{1});
                        end

                        obj.value(s.subs{:}) = varargin{1};
                    else
                        V = varargin{1};
                        %  nonColon = ~strcmp(s.subs,':');
                        %                         if sum(nonColon)==1 && obj.onehotencoding(nonColon) && isempty(V)
                        %                             % calls X(:,:,idx,:) = []; when corresponding
                        %                             % dim is one-hot-encoded
                        %    % !! should improve computational cost for
                        %
                        %                         d = find(nonColon);
                        %
                        %                         subset = s.subs{d}; % subset of values to be removed along dimension d
                        %                         if islogical(subset)
                        %                             subset = find(subset);
                        %                         end
                        %                         assert(max(subset)<=obj.size(d), 'incorrect indices');
                        %
                        %                         % construct mapping from old to new along OHE dim
                        %                         mapping = 1:obj.size(d);
                        %                         for i=1:length(subset)
                        %                             sbs = subset(i);
                        % mapping(sbs+1:end) = mapping(sbs+1:end)- 1; % if I remove element at position sbs, remove one position for all elements coming after
                        %                         end
                        %                         mapping(subset)= 0; % elements to be removed get a 0 in OHE
                        %
                        %                         % apply mapping one-hot-encoding
                        %                         obj.sub{d} = mapping(obj.sub{d});
                        %                         error('WIP');
                        %
                        %                         elseif isempty(V) % remove elements from array


                        if  isempty(V) % remove elements from array
                            if ~issparse(obj.value)
                                obj.value = reshape(obj.value, obj.siz);
                                obj.value =  subsasgn(obj.value, s, varargin{:});
                                % finish this
                                return;
                            end

                            nonColon = ~strcmp(s.subs,':');
                            if sum(nonColon)>1
                                error('A null assignment can have only one non-colon index.');
                            end
                            if ~any(nonColon) % remove all elements
                                obj = sparsearray([]);
                            end
                            d = find(nonColon);

                            if d==1 % remove along first dimension

                                obj.value = reshape(obj.value, obj.size(1), prod(obj.size(2:end))); % reshape as dim 1 x (all other dims)
                                s.subs(3:end) = [];
                                obj.value =  subsasgn(obj.value, s, varargin{:}); % remove rows
                                obj.siz(1) = size(obj.value,1);
                                obj.value = obj.value(:);

                            elseif d==obj.ndims % remove along last dimension
                                obj.value = reshape(obj.value,prod(obj.size(1:end-1)), obj.size(end)); % reshape as  (all other dims)x last dim
                                s.subs(2:end-1) = [];
                                obj.value =  subsasgn(obj.value, s, varargin{:}); % remove columns ( "X(:,idx) = [];")
                                obj.siz(d) = size(obj.value,2); % new number of elements along last dimension
                                obj.value = obj.value(:);

                            else % we remove along an intermediate dimension
                                ind = find(obj.value(:)); % indices of nonzeros
                                val = obj.value(ind); % corresponding values

                                % get corresponding indices along dim d (and group dimensions before and
                                % after)
                                size_3d = [prod(obj.size(1:d-1)), obj.size(d), prod(obj.size(d+1:end))];
                                [ifirst,i_sub, ilast] = ind2sub(size_3d, ind);

                                % subset of values to be removed along dimension d
                                subset = s.subs{d};
                                if islogical(subset)
                                    subset = find(subset);
                                end
                                subset(subset<0) = [];
                                subset(subset>obj.size(d)) = [];
                                subset = unique(subset);

                                % construct mapping from old to new along OHE dim
                                mapping = 1:obj.size(d);
                                for i=1:length(subset)
                                    sbs = subset(i);
                                    mapping(sbs+1:end) = mapping(sbs+1:end)- 1; % if I remove element at position sbs, remove one position for all elements coming after
                                end
                                mapping(subset)= 0; % elements to be removed

                                i_sub = mapping(i_sub)'; % apply mapping to positions along corresponding dim

                                % remove elements that gets mapped to 0
                                ifirst(~i_sub)= [];
                                ilast(~i_sub)= [];
                                val(~i_sub)= [];
                                i_sub(~i_sub)= [];

                                % update size
                                new_size = obj.size(d) - length(subset);  % withdraw number of deleted indices
                                size_3d(2) = new_size;
                                obj.siz(d) = new_size;

                                % rebuild indices
                                ind = sub2ind(size_3d, ifirst,i_sub, ilast);

                                % recreate sparse value vector
                                obj.value = sparse(ind,ones(size(ind)), val, prod(obj.size),1);
                            end

                            %                             % get linear indices from sub-indices
                            %                         [ind,nsub] = get_indices(obj.siz, s.subs);
                            %
                            %                             nonColon = find(obj.siz ~= nsub);
                            %                             if length(nonColon)>1
                            %                                 error('A null assignment can have only one non-colon index.');
                            %                             end
                            %
                            %                             obj.value(ind) = [];
                            %                             obj.siz(nonColon) = obj.siz(nonColon) - nsub(nonColon); % new size
                            %
                            %                             if nonColon==length(obj.siz) && obj.siz(nonColon)==1
                            %                                 obj.siz(nonColon) = []; % remove last dimension if it has become singleton
                            %                             end

                        else % assign new values to array
                            % get linear indices from sub-indices
                            [ind,nsub] = get_indices(obj.siz, s.subs);

                            if numel(V) ~= prod(nsub)
                                error('Unable to perform assignment because the size of the left and right sides do not coincide');
                            end
                            if issparse(V)
                                if isa(V, 'sparsearray')
                                    V = fullcoding(V);
                                    [II,~,VV] = find(V.value);
                                else
                                    [II, ~, VV] = find(V(:)); % find non-zero values
                                end

                                obj.value(ind(II)) = VV;
                            else

                                % assign indices
                                obj.value(ind) = varargin{1};
                            end
                        end
                    end

                    %          if length(s) == 1
                    %             % Implement obj(indices) = varargin{:};
                    %          elseif length(s) == 2 && strcmp(s(2).type,'.')
                    %             % Implement obj(indices).PropertyName = varargin{:};
                    %             ...
                    %          elseif length(s) == 3 && strcmp(s(2).type,'.') && strcmp(s(3).type,'()')
                    %             % Implement obj(indices).PropertyName(indices) = varargin{:};
                    %             ...
                    %          else
                    %             % Use built-in for any other expression
                    %             obj = builtin('subsasgn',obj,s,varargin{:});
                    %          end
                case '{}'
                    error('Brace indexing is not supported for variables of this type.');

                    %          if length(s) == 1
                    %             % Implement obj{indices} = varargin{:}
                    %             ...
                    %          elseif length(s) == 2 && strcmp(s(2).type,'.')
                    %             % Implement obj{indices}.PropertyName = varargin{:}
                    %             ...
                    %             % Use built-in for any other expression
                    %             obj = builtin('subsasgn',obj,s,varargin{:});
                    %          end
                otherwise
                    error('Not a valid indexing expression')
            end
        end

        %% SUB-ARRAY ALONG ONE DIMENSION
        function obj = sub_along_dimension(obj,subset,dim)

            % index for observations subset
            Xidx = cell(1,ndims(obj.value));
            Xidx{dim} = subset; % subset along selected dimension
            for d = setdiff(1:ndims(obj.value),dim) % for non sparse matrices
                Xidx{d} = 1:size(obj.value,d); % take all data along other dimensions
            end

            obj.value = obj.value(Xidx{:});


            %% subindex coding dimensions
            for i=1:length(obj.subs)
                if ~isempty(obj.sub{i}) && size(obj.sub{i},dim)>1

                    % index within X for observations subset
                    Xidx = cell(1,ndims(obj.sub{i}));
                    Xidx{dim} = subset;
                    for d = setdiff(1:ndims(obj.sub{i}),dim) % for non sparse matrices
                        Xidx{d} = 1:size(obj.sub{i},d); % take all data along other dimensions
                    end

                    obj.sub{i} = obj.sub{i}(Xidx{:});
                end
            end

        end

        %% ADDITION
        function obj = plus(obj,obj2)
            obj = oper('plus',obj,obj2, true);
        end

        %% SUBSTRACTION
        function obj = minus(obj,obj2)
            obj = oper('minus',obj,obj2, true);
        end
        %% ELEMENT-WISE PRODUCT
        function obj = times(obj,obj2)
            obj = oper('times',obj,obj2, false);
        end

        %% PRODUCT
        function obj = mtimes(obj1,obj2)
            Str.type = '()'; % to call obj(:)
            Str.subs = {':'};
            if isscalar(obj2) % sparsearray * scalar
                obj = subsasgn(obj1,Str, subsref(obj1,Str)*obj2);
            elseif isscalar(obj1) % scalar * sparsearray
                obj = subsasgn(obj2,Str, subsref(obj2,Str)*obj1);

            elseif ~ismatrix(obj1) || ~ismatrix(obj2)
                error('Arguments must be 2-D, or at least one argument must be scalar. Use TIMES (.*) for elementwise multiplication.');

            else
                if isa(obj1,'sparsearray')
                    obj1 = matrix(obj1);
                end
                if isa(obj2,'sparsearray')
                    obj2 = matrix(obj2);
                end
                obj = obj1*obj2;
            end

        end

        %% RIGHT ARRAY DIVISION
        function obj = rdivide(obj,obj2)
            obj = oper('rdivide',obj,obj2, true);
        end

        %% TEST EQUALITY
        function bool = eq(obj,obj2)
            %  if isscalar(obj2)
            %  Str.type = '()';
            %  Str.subs = {':'};
            %   bool = (subsref(obj,Str) == obj2);
            %      bool = sparsearray(bool);
            %      bool = reshape(bool,size(obj));
            %  else
            bool = oper('eq',obj,obj2, true);
            %  end
        end

        %% TEST INEQUALITIES
        function bool = le(obj,obj2)
            bool = oper('le',obj,obj2, true);
        end
        function bool = ge(obj,obj2)
            bool = oper('ge',obj,obj2, true);
        end
        function bool = lt(obj,obj2)
            bool = oper('lt',obj,obj2, true);
        end
        function bool = gt(obj,obj2)
            bool = oper('gt',obj,obj2, true);

        end
        function bool = ne(obj,obj2)
            bool = oper('ne',obj,obj2, true);
        end
        function bool = isnan(obj)
            bool = isnan(allvalues(obj));
            S = size(obj);
            bool = sparsearray(bool);
            bool = reshape(bool,S);

        end

        %% LOGICAL
        function obj = logical(obj)
            obj.value = logical(obj.value);
            obj.sub = cellfun(@logical, obj.sub, 'unif',0);
        end


        %% UNITARY MINUS
        function obj = uminus(obj)
            obj.value = -obj.value;
        end

        %% ABS
        function obj=abs(obj)
            obj.value = abs(obj.value);
        end

        %% CEIL, FLOOR, FIX, MOD, REM
        function obj=ceil(obj)
            obj.value = ceil(obj.value);
        end
        function obj=floor(obj)
            obj.value = floor(obj.value);
        end
        function obj=fix(obj)
            obj.value = fix(obj.value);
        end
        function obj=mod(obj,m)
            obj.value = mod(obj.value,m);
        end
        function obj=rem(obj,m)
            obj.value = rem(obj.value,m);
        end

        %% SUM OVER DIMENSION
        function obj = sum(obj,d)
            if nargin==1 % default sum over dim 1
                d = 1;
            end
            if ischar(d) && strcmp(d, 'all')
                obj = sum(allvalues(obj));
            else
                U = cell(1,ndims(obj));
                U{d} = ones(1,size(obj,d));
                obj = tensorprod(obj,U);
            end
        end

        %% MEAN OVER DIMENSION
        function obj = mean(obj,d)
            if ischar(d) && strcmp(d, 'all')
                obj = sum(obj,'all')/numel(obj);
            else
                obj = sum(obj,d)/size(obj,d);
            end
        end

        %% ALL VALUES
        function x = allvalues(obj)
            Str.type = '()'; % to call obj(:)
            Str.subs = {':'};
            x = subsref(obj,Str);
        end

        %% TENSOR PRODUCT
        function P = tensorprod(obj,U)
            %if nargin<6 % by default, squeeze resulting matrix
            %    do_squeeze = 1;
            %end
            % if nargin<7 % by default, do not collapse over observations
            %     Uobs = [];
            % end

            n = ndims(obj);
            nRow = cellfun(@(x) size(x,1), U); % new size along dimensions we project on
            OneHot = onehotencoding(obj);
            OneHotDims = find(OneHot); % dim to project on and sparse dims
            noprodDims = find(cellfun(@isempty,U)); % dimensions with no tensor product

            %% collapse first over non-one-hot-encoding coding dimensions which are not used by
            % OHE dimensions
            BoolMat = true(length(OneHotDims),n); % by default singleton doms
            for ee =1:length(OneHotDims)
                ss = size(obj.sub{OneHotDims(ee)})==1; % singleton dimensions for these regressors (i.e. onehotencoding regressor does not vary along that dimension)
                BoolMat(ee,1:length(ss)) = ss;
            end
            SingletonDim = find(all(BoolMat==1,1) & ~OneHot); % dimension all singleton across onehotencoding dimensions: let's start projecting around these ones
            SingletonDim = setdiff(SingletonDim,noprodDims);
            if ~isempty(SingletonDim)
                UU = cell(1,n);
                UU(SingletonDim)=  U(SingletonDim); % for tensor projection
                NonOneHotSize = obj.size; % size of value array
                NonOneHotSize(OneHotDims) = 1;
                obj.value = tprod(obj.value, UU,NonOneHotSize); % tensor product
                obj.siz(SingletonDim) = nRow(SingletonDim);
                U(SingletonDim) = {[]};

            end

            %% now project over OHE coding dimensions
            if isempty(OneHotDims)
                OneHotDims = zeros(1,0);
            end
            if any(setdiff(OneHotDims,noprodDims)) % if any projecting one-hot-encoding dim
                % create copy of object where OHE dimensions are reduced to
                % singleton
                V = sparsearray(obj.value);
                S = size(obj);
                SS = ones(1,ndims(obj)); % OHE dims are reduced to singleton
                SS(~onehotencoding(obj)) = S(~onehotencoding(obj));
                V = reshape(V,SS);

                for f= setdiff(OneHotDims,noprodDims) % loop through projecting one-hot-encoding dims
                    UU = [zeros(nRow(f),1) U{f}]; % projection for that OHE dimension (add 0 first for 0 index)
                    % perhaps should work on the case when U is sparse

                    % create a tensor VV with values of U that should be
                    S = size(obj.sub{f}); % size of VV, same as OHE array
                    S(end+1:f-1) = 1;
                    S(f) = nRow(f); % except for new dimension
                    VV = zeros(S); % preallocate VV

                    C = cell(1,length(S)); % cell array of indices
                    for ddd=1:length(S)
                        C{ddd} = 1:S(ddd);
                    end

                    % build iteratively for each row
                    for r=1:nRow(f)
                        uu = UU(r,:);
                        C{f} = r; % current row
                        VV(C{:}) = uu(1+obj.sub{f}); %
                    end

                    % pointwise multiply with data
                    V =  V .* VV;


                    U{f} = []; % remove projecting vector from list
                    OneHot(f) = 0; % no longer OHE dim
                    obj.sub{f} = [];
                    obj.siz(f) = nRow(f);
                end

                obj.value = matrix(reshape(V, size(V,1), numel(V)/size(V,1)));
            end

            %% deal with special case when we only need to project over one dummy dimension (should be faster this way)
            still_to_do = ~cellfun(@isempty,U);
            if sum(still_to_do)==1
                d = find(still_to_do);
                if isrow(U{d}) && all(U{d}==1) && sum(obj.onehotencoding)==1 &&...
                        size(obj.sub{obj.onehotencoding},d)>1 && nnz(obj.value)==nnz(obj.sub{obj.onehotencoding})% dummy variable for OHE
                    d_onehot = find(obj.onehotencoding);
                    [I,J,V] = find(obj.value);
                    obj.sub{d_onehot} = reshape(obj.sub{d_onehot},size(obj.value,1),size(obj.value,2));
                    [I2,J2,ind_newdim] = find(obj.sub{d_onehot});
                    if all(I==I2) && all(J==J2) % make extra sure the non-zero values coincide
                        ind_newdim = cast(ind_newdim,'like',V);
                        P = sparse(I,ind_newdim,V,obj.size(1),obj.size(d_onehot));
                        S = obj.size;
                        S(d)=1;
                        if length(S)>2
                            P = sparsearray(P);
                            P = reshape(P,S);
                            return;
                        end
                    end
                end
            end

            %% project remaining non-OHE dimensions
            obj = fullcoding(obj);

            [P,S] = tprod(obj.value, U, obj.siz); % tensor product

            if issparse(P) && length(S)>2
                P = sparsearray(P);
            end
            P = reshape(P,S);
            %            end


        end


    end
end
%% PRIVATE FUNCTIONS

%% get indices
function [ind,nSub] = get_indices(siz, OHE)
OHE = check_one_hot_encoding(siz, OHE);

nSub = cellfun(@length, OHE); % number of subindices in each dimension
nElem = prod(nSub); % total number of element of sub-array
D = length(nSub); % dimensionality

ind = ones(nElem,1);
for d=1:D
    sb = (OHE{d}(:)-1)*prod(siz(1:d-1)); % sub-indices for this dimension
    if d>1
        sb = repelem(sb, prod(nSub(1:d-1)),1); % replicate each value in this dimension according to lower dimensions
    end
    if d<D
        sb = repmat(sb,prod(nSub(d+1:end)),1); % replicate vector of subindices according to higher dimensions
    end
    ind = ind + sb;
end
end

%% check one-hot-encoding indices take correct value (for referencing and
% assignment)
function  OHE = check_one_hot_encoding(siz, OHE)

D = length(siz); % array dimension
nOneHot = length(OHE);
if nOneHot>D
    error('sparse tensor does not have so many dimensions');
elseif nOneHot<D && ~all(siz(nOneHot+1:end)==1)
    error('provide indices for all dimensions (or just one to get column output)');
else
    OHE(nOneHot+1:D) = {1};
    for d=1:D
        if ischar(OHE{d}) && OHE{d}==':'
            OHE{d} = 1:siz(d);
        elseif islogical(OHE{d})
            OHE{d} = find(OHE{d});
        end
        if any(OHE{d}<0)
            error('Array indices must be positive integers or logical values.');
        end
        if any(OHE{d}>siz(d))
            error('Index exceeds the number of array elements (%d).', siz(d));
        end
    end
end
end


%% ELEMENT-WISE OPERATION
function obj = oper(strfun,obj1,obj2, fullcode)
fun = eval(['@' strfun]);

% process the easy cases first
if  isscalar(obj2) % between sparse array and scalar
    x = fun(allvalues(obj1),obj2);
    S = size(obj1);

    obj = sparsearray(x);
    obj = reshape(obj,S);
    return;
elseif isscalar(obj1) % between scalar and sparse array
    x = fun(allvalues(obj2),obj1);
    S = size(obj2);

    obj = sparsearray(x);
    obj = reshape(obj,S);
    return;
end

% now process the hard case: operation between one sparse array and another (possibly sparse) array

% convert both to sparse arrays
obj1 = sparsearray(obj1);
obj2 = sparsearray(obj2);

if fullcode % some operations cannot work with subindex coding
    obj1 = fullcoding(obj1);
    obj2 = fullcoding(obj2);
end

% size of both objects
S1 = size(obj1);
S2 = size(obj2);
S1(end+1:length(S2)) = 1; % pad with ones if required so that they have same length
S2(end+1:length(S1)) = 1;
if ~all(S1==S2 | S1==1 | S2==1)
    error('dimensions do not match');
end

%% replicate matrices along singleton dimensions to match size
D = length(S1); % number of dimensions
R1 = ones(1,D); % replication number per dimension (default:1), for each object
R2 = ones(1,D);

fc = false(1,D); % dimensions with subindex coding
sb = cell(1,D);

for d=1:D
    if S1(d)==1 && S2(d)>1 % if obj1 is singleton is this dimension and obj2 is not
        % obj1 needs to be replicated
        if ~onehotencoding(obj2,d)
            R1(d) = S2(d);
        else % unless there is subindex coding
            fc(d) = true;
            sb{d} = obj2.sub{d};
            %  else % singleton coding in that dimension
            %      obj1.sub{d} = 1;
            %      obj1.siz(d) = S2(d);
        end
    end
    if S2(d)==1 && S1(d)>1
        if ~onehotencoding(obj1,d)
            R2(d) = S1(d);
        else
            fc(d) = true;
            sb{d} = obj1.sub{d};
            %     obj2.sub{d} = 1; % singleton coding in that dimension
            %    obj2.siz(d) = S1(d);
        end
    end
    % cannot use subindex coding if the other array is not singleton
    % along that dimension
    if S1(d)>1 && S2(d)>1
        if  onehotencoding(obj1,d)
            obj1 = fullcoding(obj1,d);
        end
        if  onehotencoding(obj2,d)
            obj2 = fullcoding(obj2,d);
        end
    end
end

S = max(S1,S2); % size of the output array
Snonsub = S;
Snonsub(fc) = 1;

if isequal(fun, @times) && prod(Snonsub)==numel(obj2.value) && all(obj1.value(:)==1)
    % if pairwise multiplication and obj1 is pure one-hot encoding, values are simply inherited from obj2
    x = obj2.value;

elseif isequal(fun, @times) && prod(Snonsub)==numel(obj1.value) && all(obj2.value(:)==1)
    % if pairwise multiplication and obj2 is pure one-hot encoding,
    % values are simply inherited from obj1
    x = obj1.value;

else
    % first replicate each object with required dimensions to equate dims
    obj1 = repmat(obj1, R1);
    obj2 = repmat(obj2, R2);

    % now apply function
    x = fun(obj1.value, obj2.value);
end

obj = sparsearray(x);
obj = reshape(obj,Snonsub);

%% add subindex coding
obj.sub = sb;
obj.siz = S;

end


%% % tensor production for array
function [X,S] = tprod(X,U,S)
% P = tensorprod(X,U)
% X: n-dimensional array
% U: cell array of vectors

if prod(S)~=numel(X)
    error('size does not match');
end

prod_dims= find(~cellfun(@isempty,U));
for d= prod_dims

    d1 = prod(S(1:d-1)); % number of elements for lower dimensions
    d2 = prod(S(d+1:end)); % number of elements for higher dimensions

    u = U{d}; % weight we project on
    nrow = size(u,1);

    if d1<d2 %select based on computationally less expensive

        new_size = [d1*S(d) d2];


        if d1>1
            u = kron(u,speye(d1));
        end

        if ~isequal(new_size, size(X))
            X  = reshape(X, new_size);
        end

        X = u*X;

    else % d2>=1

        new_size = [d1 S(d)*d2];

        u = u';
        if d2>1
            u = kron(speye(d2),u);
        end

        if ~isequal(new_size, size(X))
            X  = reshape(X, new_size);
        end

        X = X*u;

    end
    if ismatrix(X) && prod_dims(end)<=2 && issparse(u) % ideally we want the product of X*u to be directly sparse without having to go through the non-sparse version
        X = sparse(X);
    end
    S(d) = nrow;

end

end

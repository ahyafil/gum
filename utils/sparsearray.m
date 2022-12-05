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
                else
                    [I,J,V] = find(varargin{1}(:)); % find indices and values
                    obj.value = sparse(I,J,V, numel(varargin{1}),1);
                end
                obj.siz = size(varargin{1});
            end
        end

        %% DISP
        function disp(obj)
            if any(subcoding(obj))
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
            obj = fullcoding(obj);

            obj.siz = S;
            if ~issparse(obj.value) || length(S)<=2
                obj.value = reshape(obj.value,S);
            end

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

                NonSubCoding = find(~subcoding(obj));
                sz = size(obj);
                Pnon = P(ismember(P, NonSubCoding));
                new_sz = sz(Pnon); % size of non-subindex coding array

                sz = sz(NonSubCoding);


                nelem = prod(sz); % total number of element of sub-array
                D = length(NonSubCoding); % dimensionality

                ind = ones(1,nelem);
                for d=1:D
                    sb = (0:sz(d)-1)*prod(sz(1:d-1)); % sub-indices for this dimension
                    dd = find(Pnon==NonSubCoding(d)); % which position in new dimension order
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

            % permute sub-index arrays
            for d=find(subcoding(obj))
                obj.sub{d} = permute(obj.sub{d},P);
            end

            % permute order of sub-index arrays
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
            if any(subcoding(obj))
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

        %% SUBINDEX CODING (one boolean per dimension
        function bool = subcoding(obj,d)
            bool = ~cellfun(@isempty, obj.sub);
            bool(end+1:ndims(obj)) = false;
            if nargin>1
                bool = bool(d);
            end
        end

        %% ISSPARSE
        function bool = issparse(obj)
            bool = true;
        end

        %% ISSPARSEVALUE (whether field value is sparse)
        function bool = issparsevalue(obj)
            bool = issparse(obj.value);
        end

        %% FULLCODING: change subindex coding to coding
        function obj = fullcoding(obj,d)
            if nargin<2 % if dimension is not provided: for all subindex coding dimensions
                d = find(subcoding(obj));
            end
            if isempty(d)
                return;
            end
            if any(~ismember(d, find(subcoding(obj))))
                error('not a subindex coding dimension');
            end


            [Ind,~,V] = find(obj.value(:)); % indices and values of non-zero values

            NonSubDims = setdiff(1:ndims(obj), find(subcoding(obj))); % dimensions not coded with subindices
            sz = size(obj);
            sz  = sz(NonSubDims);

            Sub = cell(1,length(NonSubDims));
            [Sub{:}] = ind2sub(sz, Ind); % indices of values for non-subindex coding position
            %  SubOld = Sub;
            NSD = NonSubDims;

            for dd=d
                %  SubSub = SubOld;
                SubSub = Sub;
                for f = 1:length(NSD)
                    if size(obj.sub{dd},NSD(f)) ==1 % if subindex matrix is singleton along this dim
                        SubSub{f} = ones(size(SubSub{f}));
                    end
                end
                SubInd = sub2ind(size(obj.sub{dd}), SubSub{:}); % indices with subindex array
                NewSub = obj.sub{dd}(SubInd); % new subindex for this dimension

                rmvdata = NewSub ==0;
                if any(rmvdata)
                    for ddd = 1:length(Sub)
                        Sub{ddd}(rmvdata) = [];
                    end
                    V(rmvdata) = [];
                    NewSub(rmvdata) = [];
                end


                Sub{end+1} = NewSub;
                sz(end+1) = size(obj,dd);

                %  fct = prod(sz(NonSubDims<dd));
                %  Ind = Ind + (NewSub-1)*fct;

                obj.sub{dd} = [];
                NSD(end+1) = dd;
                %  NonSubDims = sort([NonSubDims dd]); % add to list of non sub-index dimensions
                %  sz = size(obj);
                %  sz  = sz(NonSubDims);

            end

            [NonSubDims,ord] = sort([NonSubDims d]);

            % sort dimensions
            Sub = Sub(ord);
            sz = sz(ord);

            Ind = sub2ind(sz, Sub{:});

            obj.value = sparse(Ind, ones(1,length(V)),V, prod(sz),1);
            % obj.value = reshape(obj.value, sz(1), prod(sz(2:end)));
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

            % cannot replicate sub-coding
            SubcodingDims = find(subcoding(obj));
            obj = fullcoding(obj, SubcodingDims(rep(SubcodingDims)>1));

            if issparse(obj.value) && (~ismatrix(obj) || length(rep)>2)

                NonSubCoding = ~subcoding(obj);
                sz = size(obj);
                if length(rep)>ndims(obj)
                    NonSubCoding = [NonSubCoding true(1,length(rep)-ndims(obj))];
                    sz( ndims(obj)+1:length(rep)) = 1;
                end
                rep_nonsub = rep(NonSubCoding);
                sz = sz(NonSubCoding);
                new_sz = sz .* rep_nonsub; % size of new array

                nelem = prod(new_sz); % total number of element of sub-array
                D = sum(NonSubCoding); % dimensionality

                % build the vector of indices for new matrix (this is not
                % optimal as this vector is full - should work a more
                % efficient way, perhaps converting to sparse matrix ...)
                ind = ones(1,nelem);
                for d=1:D
                    sb = (0:sz(d)-1)*prod(sz(1:d-1)); % sub-indices for this dimension
                    if rep_nonsub(d)>1 % if needs to replicate along that dimension
                        sb = repmat( sb, 1, rep_nonsub(d));
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
                obj.value = repmat(obj.value,varargin{:});
            end

            for d=find(subcoding(obj))

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

                    if ~issparse(obj.value) || length(s1.subs)==1 % obj(ind1:ind3) or obj(:)
                        if any(subcoding(obj))
                            obj= fullcoding(obj);
                            % error('not coded yet');
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

                        for d = find(~AllIndices & obj.subcoding)
                            % do not maintain one-hot encoding if referencing a subset
                            % (technically we could if there's no index repetition)
                            obj = fullcoding(obj, d);
                        end

                        if length(s1.subs)>= D && all(AllIndices(2:D))
                            %syntax: obj(v,:,:,:), only the first dim is indexed
                            S = obj.siz;

                            % get correspinding values for one-hot encoding dimensions
                            for d=find(obj.subcoding)
                                if size(obj.subs{d},1)>1
                                    obj.subs{d} = subsref(obj.subs{d}, s1);
                                end
                            end

                            % now deal with value array
                            b = reshape(obj.value,S(1),prod(S(2:D))); % reshape as matrix grouping all dimensions until first as rows

                            % select corresponding columns
                            s1.subs(3:D) = [];
                            b = subsref(b, s1);

                            % size of new array
                            nsub = S;
                            nsub(1) = size(b,1); % number of columns in b

                        elseif length(s1.subs)>= D && all(AllIndices(1:D-1))
                            %syntax: obj(:,:,:,v), only the last dim is indexed
                            S = obj.siz;

                            % get correspinding values for one-hot encoding dimensions
                            for d=find(obj.subcoding)
                                if size(obj.subs{d},D)>1
                                    obj.subs{d} = subsref(obj.subs{d}, s1);
                                end
                            end

                            % now deal with value array
                            b = reshape(obj.value,prod(S(1:D-1)),S(D)); % reshape as matrix grouping all dimensions until first as rows

                            % select corresponding columns
                            s1.subs(1:D-2) = [];
                            b = subsref(b, s1);

                            % size of new array
                            nsub = S;
                            nsub(D) = size(b,2); % number of columns in b
                        else

                            % get linear indices from sub-indices (this is very
                            % slow if indexing large number of indices
                            % irrespective of sparsity, should be improved!)

                            [ind,nsub] = get_indices(obj.siz, s1.subs);

                            %  inz = find(obj.value);

                            % extract value
                            b = obj.value(ind);
                        end

                        if length(nsub)>1
                            b = sparsearray(b);
                        end
                        b = reshape(b,nsub);

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

                        % get linear indices from sub-indices
                        [ind,nsub] = get_indices(obj.siz, s.subs);
                        V = varargin{1};

                        if isempty(V) % remove elements from array

                            nonColon = find(obj.siz ~= nsub);
                            if length(nonColon)>1
                                error('A null assignment can have only one non-colon index.');
                            end

                            obj.value(ind) = [];
                            obj.siz(nonColon) = obj.siz(nonColon) - nsub(nonColon); % new size

                            if nonColon==length(obj.siz) && obj.siz(nonColon)==1
                                obj.siz(nonColon) = []; % remove last dimension if it has become singleton
                            end

                        else % assign new values to array

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
                obj = matrix(obj1)*matrix(obj2);
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
            subcode = subcoding(obj);
            subcodeDims = find(subcode); % dim to project on and sparse dims
            noprodDims = find(cellfun(@isempty,U)); % dimensions with no tensor product

            %% collapse first over non-subindex coding dimensions which are not used by
            % subindex dimensions
            BoolMat = true(length(subcodeDims),n); % by default singleton doms
            for ee =1:length(subcodeDims)
                ss = size(obj.sub{subcodeDims(ee)})==1; % singleton dimensions for these regressors (i.e. subcoding regressor does not vary along that dimension)
                BoolMat(ee,1:length(ss)) = ss;
            end
            SingletonDim = find(all(BoolMat==1,1) & ~subcode); % dimension all singleton across subcoding dimensions: let's start projecting around these ones
            SingletonDim = setdiff(SingletonDim,noprodDims);
            if ~isempty(SingletonDim)
                UU = cell(1,n);
                UU(SingletonDim)=  U(SingletonDim); % for tensor projection
                SizeNonSubCode = obj.size; % size of value array
                SizeNonSubCode(subcodeDims) = 1;
                obj.value = tprod(obj.value, UU,SizeNonSubCode); % tensor product
                obj.siz(SingletonDim) = nRow(SingletonDim);
                U(SingletonDim) = {[]};

            end

            %% now project over subindex coding dimensions
            if isempty(subcodeDims)
                subcodeDims = zeros(1,0);
            end
            if any(setdiff(subcodeDims,noprodDims))
                V = sparsearray(obj.value);
                S = size(obj);
                SS = ones(1,ndims(obj));
                SS(~subcoding(obj)) = S(~subcoding(obj));
                V = reshape(V,SS);

                for f= setdiff(subcodeDims,noprodDims)
                    UU = [zeros(nRow(f),1) U{f}]; % weights for that sparse dimension (add 0 first if 0 index)

                    S = size(obj.sub{f});
                    S(end+1:f-1) = 1;
                    S(f) = nRow(f);
                    C = cell(1,length(S));
                    for ddd=1:length(S)
                        C{ddd} = 1:S(ddd);
                    end
                    VV = zeros(S);

                    for r=1:nRow(f)
                        uu = UU(r,:);
                        C{f} = r;
                        VV(C{:}) = uu(1+obj.sub{f});
                    end

                    V =  V .* VV;


                    %if isvector(obj.sub{f})
                    %    obj.value = obj.value .* UU(1+obj.sub{f})';
                    %else
                    %    obj.value = obj.value .* UU(1+obj.sub{f});
                    %end
                    U{f} = []; % remove this one
                    subcode(f) = 0;
                    obj.sub{f} = [];
                    obj.siz(f) = nRow(f);
                end

                obj.value = matrix(reshape(V, size(V,1), numel(V)/size(V,1)));


            end

            %             %% if any dimension with no product is sub-index coding
            %             if ~isempty(noprodDims) && any(subcode(noprodDims))
            %                 dd = noprodDims(subcode(noprodDims)); % sub-index coding dim
            %                 dd = dd(1); % let's work with the first one first
            %                 % otherd = setdiff(noprodDims,dd); % other dimensions
            %                 mm = size(U{dd},2); % size along this dimension
            %
            %                 % allocate matrix P
            %                 sizeP = ones(1,n);
            %                 sizeP(noprodDims) = size(U{noprodDims},2);% non-singleton dimensions only where we project
            %                 if isempty(U{1})
            %                     sizeP(1) = obj.siz(1);
            %                 end
            %                 P = zeros(sizeP);
            %
            %                 % spd2 = subcode;
            %                 % spd2(dd) = false;
            %
            %                 % index for projected matrix
            %                 Sub = obj.sub{dd};
            %                 nonSingleton = setdiff(1:ndims(Sub),find(size(Sub)==1)); % number of non-singleton dimensions for the subindex array
            %                 for o=1:mm %% for each value along that dimension
            %                     U2 = U;
            %                     %  Uobs2 = Uobs;
            %                     nonnull = any(Sub(:,:)==o,2); % check which observations have this value of the regressor
            %
            %                     if any(nonnull)
            %                         for q=1:length(sizeP), idx{q} = 1:sizeP(q); end
            %                        % if ~all(nonnull) % select subset of values of dimension 1
            %                        %     obj3 = sub_along_dimension(obj,nonnull,1);
            %                        % else
            %                        %     obj3 = obj;
            %                        % end
            %                         % if isempty(U{1})
            %                         %     idx{1} = nonnull; % select group of observations in P
            %                         % else
            %                         %     Uobs2 = Uobs(nonnull); % set weight vector for observations
            %                         % end
            %
            %                         % check and select other dimensions of stim that have
            %                         % non-zero values for this value of the regressor
            %                         anyt = any(obj3.sub{dd}==o,1); % any non-null value for the observation
            %                         for pp = nonSingleton % for all non-singleton dimensions (except first one that we just did)
            %                             nonnull = anyt;
            %                             for qq=setdiff(nonSingleton,[1 pp])
            %                                 nonnull = any(nonnull,qq);
            %                             end
            %                             nonnull = nonnull(:); % row vector
            %                             if ~all(nonnull)
            %                                 if any(pp-1==noprodDims)
            %                                     idx{pp} = nonnull;
            %                                 end
            %
            %                                 % select only non-null data
            %                                 obj3 = sub_along_dimension(obj,nonnull,pp);
            %                                 U2{pp} = U2{pp}(:,nonnull);
            %                                 if sum(nonnull) ==1 % if we're left with single weight
            %                                     obj3.value = obj3.value*U2{pp};
            %                                     obj3.size(pp) = nRow(pp);
            %                                     U2{pp} = [];
            %                                 end
            %                             end
            %                         end
            %                         obj3.value = obj3.value .* (obj3.sub{dd} ==o); % all data points corresponding to this index
            %                         obj3.sub{dd} = []; % remove corresponding regressor from here
            %                         obj3.size(dd) = nRow(dd);
            %
            %                         if nRow(dd) ==1
            %                            U2{dd} = [];
            %                            obj3.value = U{dd}(o) *obj3.value;
            %                         else
            %                                 U2{dd} = U{dd}(:,o);
            %                         end
            %                         idx{dd} = o;
            %                         if all(cellfun(@isempty, U2))
            %                             % if ~isempty(Uobs)
            %                             %     obj3{1} = Uobs2*obj3{1};
            %                             % end
            %                             P(idx{:}) = obj3.value;
            %
            %                         else
            %                             P(idx{:}) = tensorprod(obj3,U2);  % now perform product with other dimensions
            %                         end
            %                     end
            %                 end
            %
            %             else
            %                %% all projection dimensions are non-subindex coding

            obj = fullcoding(obj);

            % project non-sparse index dimensions
            % UU =  U; %[{Uobs} U(r,:)]; % for tensor projection
            % for f=subcodeDims
            %     UU{f} = [];
            % end
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


function [ind,nsub] = get_indices(siz, subs)
subs = check_subs(siz, subs);

nsub = cellfun(@length, subs); % number of subindices in each dimension
nelem = prod(nsub); % total number of element of sub-array
D = length(nsub); % dimensionality

ind = ones(nelem,1);
for d=1:D
    sb = (subs{d}(:)-1)*prod(siz(1:d-1)); % sub-indices for this dimension
    if d>1
        sb = repelem(sb, prod(nsub(1:d-1)),1); % replicate each value in this dimension according to lower dimensions
    end
    if d<D
        sb = repmat(sb,prod(nsub(d+1:end)),1); % replicate vector of subindices according to higher dimensions
    end
    ind = ind + sb;
end
end


%% check subs indices take correct value (for referencing and
% assignment)
function  subs = check_subs(siz, subs)

D = length(siz); % array dimension
if length(subs)>D
    error('sparse tensor does not have so many dimensions');
elseif length(subs)<D
    error('provide indices for all dimensions (or just one to get column output)');
else
    for d=1:D
        if ischar(subs{d}) && subs{d}==':'
            subs{d} = 1:siz(d);
        elseif islogical(subs{d})
            subs{d} = find(subs{d});
        end
        if any(subs{d}<0)
            error('Array indices must be positive integers or logical values.');
        end
        if any(subs{d}>siz(d))
            error('Index exceeds the number of array elements (%d).', siz(d));
        end
    end
end
end


%% ELEMENT-WISE OPERATION
function obj = oper(strfun,obj1,obj2, fullcode)
fun = eval(['@' strfun]);

if  isscalar(obj2) % between sparse array and scalar
    x = fun(allvalues(obj1),obj2);
    S = size(obj1);

    obj = sparsearray(x);
    obj = reshape(obj,S);
elseif isscalar(obj1) % between scalar and sparse array
    x = fun(allvalues(obj2),obj1);
    S = size(obj2);

    obj = sparsearray(x);
    obj = reshape(obj,S);
else % between one sparse array and another (possibly sparse) array

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
    % D = max(ndims(obj1),ndims(obj2)); % number of dimensions
    D = length(S1); % number of dimensions
    R1 = ones(1,D); % replication number per dimension (default:1), for each object
    R2 = ones(1,D);

    fc = false(1,D); % dimensions with subindex coding
    sb = cell(1,D);

    for d=1:D
        if S1(d)==1 && S2(d)>1 % if obj1 is singleton is this dimension and obj2 is not
            % obj1 needs to be replicated
            if ~subcoding(obj2,d)
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
            if ~subcoding(obj1,d)
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
            if  subcoding(obj1,d)
                obj1 = fullcoding(obj1,d);
            end
            if  subcoding(obj2,d)
                obj2 = fullcoding(obj2,d);
            end
        end
    end

    S = max(S1,S2); % size of the output array
    Snonsub = S;
    Snonsub(fc) = 1;

    if isequal(fun, @times) && all(obj1.value(:)==1)
        % if pairwise multiplication and obj1 is pure one-hot encoding, values are simply inherited from obj2
        x = obj2.value;

    elseif isequal(fun, @times) && all(obj2.value==1)
        % if pairwise multiplication and obj2 is pure one-hot encoding,
        % values are simply inherited from obj1
        x = obj1.value;

    else
        % first replicate each object with required dimensions

        % dim_R1 = S1==1 & S2>1;
        % R1(dim_R1) = S2(dim_R1);
        obj1 = repmat(obj1, R1);

        %  dim_R2 = S2==1 & S1>1;
        %  R2(dim_R2) = S1(dim_R2);
        obj2 = repmat(obj2, R2);

        %S = size(obj1);
        x = fun(obj1.value, obj2.value);
        % x = fun(allvalues(obj1),allvalues(obj2));
    end

    obj = sparsearray(x);
    obj = reshape(obj,Snonsub);

    %% add subindex coding
    obj.sub = sb;
    obj.siz = S;

end



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
    S(d) = nrow;

end

end

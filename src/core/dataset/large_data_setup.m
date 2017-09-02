function imdb = large_data_setup(varargin)

opts.dataDir = fullfile('data','AnimalFace') ;
opts.lite = false ;
opts.subCat = [];
opts.train = 0.8;
opts.test = 0.2;
opts.val = 0;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------.------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

descrs = [];
descrs_list = dir([opts.dataDir, filesep, '*']);
for i = 1:length(descrs_list)
    if descrs_list(i).name(1) ~= '.' && descrs_list(i).isdir
        descrs{end+1} = descrs_list(i).name;
    end
end


imdb.classes.name = descrs ;
imdb.classes.description = descrs ;
imdb.imageDir = opts.dataDir;

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------

fprintf('searching training and testing images ...\n') ;
train_names = {} ;
train_labels = {} ;
val_names = {} ;
val_labels = {} ;
test_names = {} ;
test_labels = {} ;
lab = 1;
for d = dir(fullfile(opts.dataDir, '*'))'
    if d.name(1) == '.' || ~d.isdir
        continue;
    end
    if ~isempty(opts.subCat) && ismember(d.name, opts.subCat) == 0
        continue;
    end
    
    ims = dir(fullfile(opts.dataDir, d.name, '*.jpg'));
    
    idx = randperm(numel(ims));
    idx_train = idx(1:max(1, floor(numel(idx) * opts.train)));
    idx_test = idx(numel(idx_train)+1 : floor(numel(idx) * (opts.train + opts.test)));
    idx_val = idx(numel(idx_train) + numel(idx_test) + 1: end);
    % training images
    train_names{end+1} = strcat([d.name, filesep], {ims(idx_train).name}) ;
    train_labels{end+1} = ones(1, numel(idx_train)) * lab ;
    % testing images
    test_names{end+1} = strcat([d.name, filesep], {ims(idx_test).name}) ;
    test_labels{end+1} = ones(1, numel(idx_test)) * lab ;
    % val images
    val_names{end+1} = strcat([d.name, filesep], {ims(idx_val).name}) ;
    val_labels{end+1} = ones(1, numel(idx_val)) * lab ;
    
    fprintf('.') ;
    if mod(numel(train_names), 50) == 0, fprintf('\n') ; end
    lab = lab + 1;
    %fprintf('found %s with %d images\n', d.name, numel(ims)) ;
end


% train
train_names = horzcat(train_names{:}) ;
train_labels = horzcat(train_labels{:}) ;

imdb.images.id = 1:numel(train_names) ;
imdb.images.name = train_names ;
imdb.images.set = ones(1, numel(train_names)) ;
imdb.images.label = train_labels;

% val
val_names = horzcat(val_names{:}) ;
val_labels = horzcat(val_labels{:}) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(val_names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, val_names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(val_names))) ;
imdb.images.label = horzcat(imdb.images.label, val_labels) ;

% test
test_names = horzcat(test_names{:}) ;
test_labels = horzcat(test_labels{:}) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(test_names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, test_names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(test_names))) ;
imdb.images.label = horzcat(imdb.images.label, test_labels) ;

% -------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------


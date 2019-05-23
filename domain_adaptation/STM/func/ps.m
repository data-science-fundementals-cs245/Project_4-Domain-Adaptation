function val = ps(param, field, defaultVal)

if isfield(param,field)
  val = param.(field);
else
  val = defaultVal;
end
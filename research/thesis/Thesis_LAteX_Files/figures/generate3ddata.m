% Data generation
xData = -1:0.1:1;
yData = -1:0.1:1;
zData = zeros(length(xData),length(yData));
for i = 1:length(xData)
  for j = 1:length(yData)
    zData(i,j) = exp(-xData(i)^2-yData(j)^2);
  end
end

% Writing file
fid = fopen("datatable3d.dat","w");
fprintf(fid,"X Y Z\n");
for i = 1:length(xData)
  for j = 1:length(yData)
    fprintf(fid, "%f %f %f\n",xData(i),yData(j),zData(i,j));
  end
  fprintf(fid,"\n");
end
fclose(fid);

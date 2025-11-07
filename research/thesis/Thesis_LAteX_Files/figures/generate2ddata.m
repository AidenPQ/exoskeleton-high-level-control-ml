% Data generation
xData = 0:0.1:4*pi;
yData1 = sin(xData);
yData2 = cos(xData);
yData3 = 0.5*ones(length(xData));

% Writing file
fid = fopen("datatable2d.dat","w");
fprintf(fid,"X Y1 Y2 Y3\n");
for i = 1:length(xData)
  fprintf(fid, "%f %f %f %f\n",xData(i),yData1(i),yData2(i),yData3(i));
end
fclose(fid);

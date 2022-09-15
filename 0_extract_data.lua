require 'image'
require 'torch'
require 'lib/kitti_tools'
require 'xlua'

cmd = torch.CmdLine()
cmd:option('-size', 7480, 'number of images loaded')
opt = opt or cmd:parse(arg or {})

function _typeToNumber(imgType)
 tmResult = 0 
 if imgType == 'Car' then
   tmResult = 1 
 end
 if imgType == 'Van' then
   tmResult = 2 
 end
 if imgType == 'Tram' then
   tmResult = 3
 end
 if imgType == 'Cyclist' then
   tmResult = 4 
 end
 if imgType == 'Pedestrian' then
   tmResult = 5 
 end
 if imgType == 'Person_sitting' then
   tmResult = 6
 end
 if imgType == 'Misc' then
   tmResult = 7
 end
 if imgType == 'Truck' then
   tmResult = 8
 end
 if imgType == 'DontCare' then
   tmResult = 9
 end
 return tmResult
end


local patch_w = 32
local patch_h = 32
trainData = {
  	      data = torch.DoubleTensor(80256,3,patch_w,patch_h),
   	      labels = torch.LongStorage(80265):fill(0), 
              occluded = torch.LongStorage(80265):fill(0)
	   }

local cntDt = 0
print('Read images')
for i =0, opt.size do
  xlua.progress(i, opt.size)
  local img = read_image('data/images/training/image_2' , i)
  local lbltbl = read_labels('data/labels/label_2',i)
  for j = 1,table.getn(lbltbl) do
    
    if (lbltbl[j].x2 > lbltbl[j].x1) and (lbltbl[j].y2 > lbltbl[j].y1) then
      if (lbltbl[j].x2 < img:size(3)) and (lbltbl[j].x1 < img:size(3)) and 
        (lbltbl[j].y2 < img:size(2)) and ( lbltbl[j].y1 < img:size(2)) then

        local imgSub = image.crop(img,lbltbl[j].x1,lbltbl[j].y1,lbltbl[j].x2,lbltbl[j].y2)
        yDiff = imgSub:size(2)
        xDiff = imgSub:size(3) 

        local emptyImgSub = torch.DoubleTensor(imgSub:size(1), math.max(xDiff,yDiff), math.max(xDiff,yDiff)):fill(1)

        if yDiff > xDiff then
           emptyImgSub = image.rgb2yuv(emptyImgSub)    
           imgSub = image.rgb2yuv(imgSub)
           print(emptyImgSub[{{},{1,yDiff},{((yDiff-xDiff) / 2) + 1 ,(yDiff + xDiff) /2}}] :size())
           print(imgSub:size())
           emptyImgSub[{{},{1,yDiff},{((yDiff-xDiff) / 2) + 1 ,(yDiff + xDiff) /2}}] = imgSub
        elseif xDiff > yDiff then
           emptyImgSub = image.rgb2yuv(emptyImgSub)    
           imgSub = image.rgb2yuv(imgSub)
           
           print(emptyImgSub[{{},{(xDiff-yDiff) / 2 ,((xDiff-yDiff)/2)+yDiff },{1,xDiff}}]:size())
          print( imgSub:size())
           emptyImgSub[{{},{((xDiff-yDiff) / 2) +1 ,(xDiff+yDiff)/2 },{1,xDiff}}] = imgSub
        end
        emptyImgSub = image.scale(emptyImgSub,patch_w,patch_h)
        imgSlbl = _typeToNumber(lbltbl[j].type)
        cntDt = cntDt + 1
        trainData.data[cntDt] = emptyImgSub 
        trainData.labels[cntDt] = imgSlbl
        trainData.occluded[cntDt] = lbltbl[j].occluded 
      end
    end
  end
end

trainData.data = trainData.data[{{1,cntDt},{},{}}]
tmoccluded = trainData.occluded
tmlabels = trainData.labels

trainData.labels = torch.LongStorage(cntDt):fill(0)
trainData.occluded = torch.LongStorage(cntDt):fill(0)

print('Write images in t7 format')
for i = 1, cntDt do
  xlua.progress(i, cntDt)
  trainData.labels[i] = tmlabels[i]
  trainData.occluded[i] = tmoccluded[i]     
end
torch.save('extracted_data_yuv.t7',trainData)

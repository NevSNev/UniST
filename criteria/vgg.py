from torch import nn

class encoder5(nn.Module):
    def __init__(self):
        super(encoder5,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,512,3,1,0)
        self.relu11 = nn.ReLU(inplace=True)

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(512,512,3,1,0)
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(512,512,3,1,0)
        self.relu13 = nn.ReLU(inplace=True)

        self.maxPool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(512,512,3,1,0)
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self,x,sF=None,contentV256=None,styleV256=None,matrix11=None,matrix21=None,matrix31=None):
        output = {}
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        output['r11'] = self.relu2(out)
        out = self.reflecPad7(output['r11'])

        #out = self.reflecPad3(output['r11'])
        out = self.conv3(out)
        output['r12'] = self.relu3(out)

        output['p1'] = self.maxPool(output['r12'])
        out = self.reflecPad4(output['p1'])
        out = self.conv4(out)
        output['r21'] = self.relu4(out)
        out = self.reflecPad7(output['r21'])

        #out = self.reflecPad5(output['r21'])
        out = self.conv5(out)
        output['r22'] = self.relu5(out)

        output['p2'] = self.maxPool2(output['r22'])
        out = self.reflecPad6(output['p2'])
        out = self.conv6(out)
        output['r31'] = self.relu6(out)
        if(styleV256 is not None):
            feature = matrix31(output['r31'],sF['r31'],contentV256,styleV256)
            out = self.reflecPad7(feature)
        else:
            out = self.reflecPad7(output['r31'])
        out = self.conv7(out)
        output['r32'] = self.relu7(out)

        out = self.reflecPad8(output['r32'])
        out = self.conv8(out)
        output['r33'] = self.relu8(out)

        out = self.reflecPad9(output['r33'])
        out = self.conv9(out)
        output['r34'] = self.relu9(out)

        output['p3'] = self.maxPool3(output['r34'])
        out = self.reflecPad10(output['p3'])
        out = self.conv10(out)
        output['r41'] = self.relu10(out)

        out = self.reflecPad11(output['r41'])
        out = self.conv11(out)
        output['r42'] = self.relu11(out)

        out = self.reflecPad12(output['r42'])
        out = self.conv12(out)
        output['r43'] = self.relu12(out)

        out = self.reflecPad13(output['r43'])
        out = self.conv13(out)
        output['r44'] = self.relu13(out)

        output['p4'] = self.maxPool4(output['r44'])

        out = self.reflecPad14(output['p4'])
        out = self.conv14(out)
        output['r51'] = self.relu14(out)
        return output
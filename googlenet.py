def googlenet(bs):
    return [[50176 * bs, 192, 27],
            [50176 * bs, 64, 192],
            [50176 * bs, 96, 192],
            [50176 * bs, 128, 864],
            [50176 * bs, 16, 192],
            [50176 * bs, 32, 144],
            [50176 * bs, 32, 288],
            [50176 * bs, 32, 192],
            [50176 * bs, 128, 256],
            [50176 * bs, 192, 1152],
            [50176 * bs, 32, 256],
            [50176 * bs, 96, 288],
            [50176 * bs, 96, 864],
            [50176 * bs, 64, 256],
            [12544 * bs, 192, 480],
            [12544 * bs, 96, 480],
            [12544 * bs, 208, 864],
            [12544 * bs, 16, 480],
            [12544 * bs, 48, 144],
            [12544 * bs, 48, 432],
            [12544 * bs, 64, 480],
            [12544 * bs, 160, 512],
            [12544 * bs, 112, 512],
            [12544 * bs, 224, 1008],
            [12544 * bs, 24, 512],
            [12544 * bs, 64, 216],
            [12544 * bs, 64, 576],
            [12544 * bs, 64, 512],
            [12544 * bs, 128, 512],
            [12544 * bs, 256, 1152],
            [12544 * bs, 144, 512],
            [12544 * bs, 288, 1296],
            [12544 * bs, 32, 512],
            [12544 * bs, 64, 288],
            [12544 * bs, 256, 528],
            [12544 * bs, 160, 528],
            [12544 * bs, 320, 1440],
            [12544 * bs, 32, 528],
            [12544 * bs, 128, 288],
            [12544 * bs, 128, 1152],
            [12544 * bs, 128, 528],
            [3136 * bs, 256, 832],
            [3136 * bs, 160, 832],
            [3136 * bs, 320, 1440],
            [3136 * bs, 32, 832],
            [3136 * bs, 128, 288],
            [3136 * bs, 128, 1152],
            [3136 * bs, 128, 832],
            [3136 * bs, 384, 832],          
            [3136 * bs, 192, 832],
            [3136 * bs, 384, 1728],
            [3136 * bs, 48, 832],
            [3136 * bs, 128, 432]
        ]

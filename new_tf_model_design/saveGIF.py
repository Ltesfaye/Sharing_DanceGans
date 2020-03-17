from PIL import Image, ImageDraw

def list_pillow_images_to_gif(images,name):
    images[0].save('./gengifs/'+name+'.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

def form_image_frame(keyPoints,height=700,width=1280):
    im = Image.new('RGB', (width, height), (0, 0, 0)) 
    draw = ImageDraw.Draw(im) 
    p56 = [(keyPoints[2*5]+keyPoints[2*6])/2., ( keyPoints[2*5+1]+keyPoints[2*6+1])/2.]
    p11_12 = [(keyPoints[2*11]+keyPoints[2*12])/2., ( keyPoints[2*11+1]+keyPoints[2*12+1])/2.]

    keyPoints += p56
    keyPoints += p11_12

    pairs =[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[0,17],[17,18],[11,12],[11,13],[13,15],[12,14],[14,16]]
    # draw.point(keyPoints,fill=(255,255,255))
    for i in range(17):
        x = keyPoints[2*i]
        y = keyPoints[2*i+1]
        draw.ellipse((x-2.5, y-2.5, x+2.5, y+2.5), fill = 'green', outline ='green')

    for p in pairs:
        x0= keyPoints[2*p[0]]
        y0=keyPoints[2*p[0]+1]
        x1 = keyPoints[2*p[1]]
        y1 = keyPoints[2*p[1]+1]
        draw.line((x0,y0, x1,y1), fill=(255,255,255), width=6)



    return im



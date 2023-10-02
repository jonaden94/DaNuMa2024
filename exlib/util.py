def back_to_original(image):
    image = image * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = image + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    image = image * 255
    image = image.numpy()
    image = image.astype('uint8')
    image = np.transpose(image, [1, 2, 0])
    return image
    
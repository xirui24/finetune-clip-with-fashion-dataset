import json
import torch
from matplotlib import pyplot as plt

class SaveModel:
    """
    Class to save the best model while training. 
    If the current epoch's validation loss is less than the previous least less, 
    then save the model state.
    """
    def __init__(self, dir, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.dir = dir
        
    def __call__(self, current_valid_loss, epoch, model, processor, best=True):
        if best == True:
            if current_valid_loss < self.best_valid_loss:
                self.best_valid_loss = current_valid_loss
                print(f"Best validation loss: {self.best_valid_loss}")
                print(f"Saving best model for epoch: {epoch}\n")
                model.save_pretrained(f'{self.dir}best')
                processor.save_pretrained(f'{self.dir}best')
        else:
            print("Saving the last model")
            model.save_pretrained(f'{self.dir}last')
            processor.save_pretrained(f'{self.dir}last')


class Logger():
    def __init__(self, save_dir, bs, lr, wd, r, lora_alpha, lora_dropout, bias, lora_target_modules):
        self.save_dir = save_dir
        self.log = [{"batch_size":bs, 
                    "learning_rate":lr, 
                    "weight_decay":wd,
                    "r":r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "lora_bias":bias,
                    "lora_target_modules":lora_target_modules}]

    def add_item(self, item):
        self.log.append(item)
    
    def save_log(self):
        with open(self.save_dir+"log.json", 'w', encoding='utf-8') as fp:
            fp.write(
                '[' +
                ',\n'.join(json.dumps(i,ensure_ascii=False) for i in self.log) +
                ']\n')


def train_and_validate(model, processor, train_dataloader, val_dataloader, optimizer, scheduler, save_model, epoch, batch_size, device, result_dir, logger, tolerance):
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = 999999
    trigger_time = 0

    for epoch in range(epoch):
        print(f"================= epoch {epoch} =================")
        epoch_train_losses = []
        epoch_train_accuracies = []
        epoch_val_losses = []
        epoch_val_accuracies = []

        # ===========================training===========================
        model.train()
        print("== training ==")
        for(text,image) in train_dataloader.make_batchs(batch_size) :
            optimizer.zero_grad()

            inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], attention_mask=inputs["attention_mask"], return_loss=True)

            # prediction
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            logits_per_text = outputs.logits_per_text

            logits_per_image = logits_per_image.to(device)
            logits_per_text = logits_per_text.to(device)

            probs = logits_per_text.softmax(dim=1)
            ground_truth = torch.arange(len(text),dtype=torch.long,device=device)

            # loss computation
            loss = outputs.loss
            loss.backward()
            epoch_train_losses.append(loss.item())
            
            # accuracy computation
            predictions = probs.argmax(axis=0)
            accuracy = (predictions==ground_truth).sum()/len(text)
            epoch_train_accuracies.append(accuracy)

            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
                
        epoch_train_loss = sum(epoch_train_losses)/len(epoch_train_losses)
        train_losses.append(epoch_train_loss)
        print("train loss", epoch_train_loss)

        epoch_train_acc = sum(epoch_train_accuracies)/len(epoch_train_accuracies)
        epoch_train_acc = float(epoch_train_acc)
        train_accuracies.append(epoch_train_acc)
        print("train accuracy", epoch_train_acc)

        # ===========================validation====================================
        print("== validation ==")

        model.eval()
        for(text,image) in val_dataloader.make_batchs(batch_size) :

            with torch.no_grad():
                inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
                outputs = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], attention_mask=inputs["attention_mask"], return_loss=True)
                
                # prediction
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                logits_per_text = outputs.logits_per_text

                logits_per_image = logits_per_image.to(device)
                logits_per_text = logits_per_text.to(device)

                probs = logits_per_text.softmax(dim=1)
                ground_truth = torch.arange(len(text),dtype=torch.long,device=device)

                # loss computation
                loss = outputs.loss
                epoch_val_losses.append(loss.item())

                # accuracy computation
                predictions = probs.argmax(axis=0)
                accuracy = (predictions==ground_truth).sum()/len(image)
                epoch_val_accuracies.append(accuracy)

                torch.cuda.empty_cache()

        epoch_val_loss = sum(epoch_val_losses)/len(epoch_val_losses)
        val_losses.append(epoch_val_loss)
        print("val loss", epoch_val_loss)

        epoch_val_acc = sum(epoch_val_accuracies)/len(epoch_val_accuracies)
        epoch_val_acc = float(epoch_val_acc)
        val_accuracies.append(epoch_val_acc)
        print(f"val accuracy {epoch_val_acc}\n")

        save_model(epoch_val_loss, 
                        epoch, 
                        model, 
                        processor)

        logger.add_item({"Epoch":epoch, "train_loss":epoch_train_loss, "train_acc":epoch_train_acc, "val_loss":epoch_val_loss, "val_acc":epoch_val_acc})

        # early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            trigger_time = 0

        else:
            trigger_time += 1
            if trigger_time >= tolerance:
                print(f"Early stopping at epoch {epoch}.")
                break

    # save the last model
    save_model(epoch_val_loss, 
                        epoch, 
                        model, 
                        processor,
                        best=False)
    
    logger.save_log()

    plot_result("loss", train_losses, val_losses, result_dir)
    plot_result("accuracy", train_accuracies, val_accuracies, result_dir)



def plot_result(metric_type, train_result, val_result, result_dir):
    fig = plt.figure()
    plt.plot(train_result, label = "train")
    plt.plot(val_result, label='val')
    plt.xlabel("Epoch")
    plt.ylabel(metric_type)
    plt.legend()
    plt.savefig(result_dir+metric_type+".png")
    plt.close(fig)
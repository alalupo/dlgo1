from dlgo.data.parallel_processor import GoDataProcessor


def main():
    processor = GoDataProcessor()
    generator = processor.load_go_data('train', 100, use_generator=True)
    batch_size = 10
    num_classes = 19*19
    num_samples = generator.get_num_samples(batch_size=batch_size, num_classes=num_classes)
    print(num_samples)
    data_gen = generator.generate(batch_size=batch_size, num_classes=num_classes)
    for i in range(num_samples // batch_size):
        X, y = next(data_gen)


if __name__ == '__main__':
    main()

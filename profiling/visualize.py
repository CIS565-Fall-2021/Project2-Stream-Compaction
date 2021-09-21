import csv
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.defchararray import array


def process_profile_raw():
  file = open('profile_raw.txt')

  # initialize arrays
  array_size = []  # in log2 scale
  cpu_scan_power_of_two = []
  cpu_scan_non_power_of_two = []
  naive_scan_power_of_two = []
  naive_scan_non_power_of_two = []
  work_efficient_scan_power_of_two = []
  work_efficient_scan_non_power_of_two = []
  thrust_scan_power_of_two = []
  thrust_scan_non_power_of_two = []
  cpu_compact_power_of_two = []
  cpu_compact_non_power_of_two = []
  cpu_compact_scan = []
  work_efficient_compact_power_of_two = []
  work_efficient_compact_non_power_of_two = []

  while True:
    line = file.readline()
    if not line:
      break
    array_size.append(int(line))
    cpu_scan_power_of_two.append(float(file.readline()))
    cpu_scan_non_power_of_two.append(float(file.readline()))
    naive_scan_power_of_two.append(float(file.readline()))
    naive_scan_non_power_of_two.append(float(file.readline()))
    work_efficient_scan_power_of_two.append(float(file.readline()))
    work_efficient_scan_non_power_of_two.append(float(file.readline()))
    thrust_scan_power_of_two.append(float(file.readline()))
    thrust_scan_non_power_of_two.append(float(file.readline()))
    cpu_compact_power_of_two.append(float(file.readline()))
    cpu_compact_non_power_of_two.append(float(file.readline()))
    cpu_compact_scan.append(float(file.readline()))
    work_efficient_compact_power_of_two.append(float(file.readline()))
    work_efficient_compact_non_power_of_two.append(float(file.readline()))

  # write to CSV for better data management
  with open("profile.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'Array Size', 'CPU Scan (power of two)', 'CPU Scan (non power of two)',
        'Naive Scan (power of two)', 'Naive Scan (non power of two)',
        'Work Efficient Scan (power of two)',
        'Work Efficient Scan (non power of two)', 'Thrust Scan (power of two)',
        'Thrust Scan (non power of two)',
        'CPU Stream Compaction without Scan (power of two)',
        'CPU Stream Compaction without Scan (non power of two)',
        'CPU Stream Compaction with Scan',
        'Work Efficient Stream Compaction (power of two)',
        'Work Efficient Stream Compaction (non power of two)'
    ])
    csv_writer.writerows(
        zip(array_size, cpu_scan_power_of_two, cpu_scan_non_power_of_two,
            naive_scan_power_of_two, naive_scan_non_power_of_two,
            work_efficient_scan_power_of_two,
            work_efficient_scan_non_power_of_two, thrust_scan_power_of_two,
            thrust_scan_non_power_of_two, cpu_compact_power_of_two,
            cpu_compact_non_power_of_two, cpu_compact_scan,
            work_efficient_compact_power_of_two,
            work_efficient_compact_non_power_of_two))

  return (array_size, cpu_scan_power_of_two, cpu_scan_non_power_of_two,
          naive_scan_power_of_two, naive_scan_non_power_of_two,
          work_efficient_scan_power_of_two,
          work_efficient_scan_non_power_of_two, thrust_scan_power_of_two,
          thrust_scan_non_power_of_two, cpu_compact_power_of_two,
          cpu_compact_non_power_of_two, cpu_compact_scan,
          work_efficient_compact_power_of_two,
          work_efficient_compact_non_power_of_two)


def main():
  # varying block size
  efficient_block_size = [1024, 512, 256, 128, 64, 32]  # pick: 128
  efficient_time = [0.714144, 0.78016, 1.33078, 0.275328, 0.48048, 0.402848]

  naive_block_size = [1024, 512, 256, 128, 64, 32]  # pick: 256
  naive_time = [0.276192, 0.24032, 0.21904, 0.28976, 0.228416, 0.33616]

  (array_size, cpu_scan_power_of_two, cpu_scan_non_power_of_two,
   naive_scan_power_of_two, naive_scan_non_power_of_two,
   work_efficient_scan_power_of_two, work_efficient_scan_non_power_of_two,
   thrust_scan_power_of_two, thrust_scan_non_power_of_two,
   cpu_compact_power_of_two, cpu_compact_non_power_of_two, cpu_compact_scan,
   work_efficient_compact_power_of_two,
   work_efficient_compact_non_power_of_two) = process_profile_raw()

  # visualization
  # power-of-two
  plt.figure()
  plt.plot(
      np.array(array_size[4:]), np.log10(np.array(cpu_scan_power_of_two[4:])),
      '.-')
  plt.plot(
      np.array(array_size[4:]), np.log10(np.array(naive_scan_power_of_two[4:])),
      '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(work_efficient_scan_power_of_two[4:])), '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(thrust_scan_power_of_two[4:])), '.-')
  plt.legend(['CPU', 'Naive', 'Work Efficient', 'Thrust'])
  plt.xticks(array_size[4:])
  plt.xlabel('Array Size [Log2 Scale]')
  plt.ylabel('Time [Log10 ms]')
  plt.title('Scan Runtime vs. Array Size (power-of-two)')

  # non-power-of-two
  plt.figure()
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(cpu_scan_non_power_of_two[4:])), '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(naive_scan_non_power_of_two[4:])), '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(work_efficient_scan_non_power_of_two[4:])), '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(thrust_scan_non_power_of_two[4:])), '.-')
  plt.legend(['CPU', 'Naive', 'Work Efficient', 'Thrust'])
  plt.xticks(array_size[4:])
  plt.xlabel('Array Size [Log2 Scale]')
  plt.ylabel('Time [Log10 ms]')
  plt.title('Scan Runtime vs. Array Size (non-power-of-two)')

  # compact, power-of-two
  plt.figure()
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(cpu_compact_power_of_two[4:])), '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(work_efficient_compact_power_of_two[4:])), '.-')
  plt.legend(['CPU', 'Work Efficient'])
  plt.xticks(array_size[4:])
  plt.xlabel('Array Size [Log2 Scale]')
  plt.ylabel('Time [Log10 ms]')
  plt.title('Compaction Runtime vs. Array Size (power-of-two)')

  # compact, non-power-of-two
  plt.figure()
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(cpu_compact_non_power_of_two[4:])), '.-')
  plt.plot(
      np.array(array_size[4:]),
      np.log10(np.array(work_efficient_compact_non_power_of_two[4:])), '.-')
  plt.legend(['CPU', 'Work Efficient'])
  plt.xticks(array_size[4:])
  plt.xlabel('Array Size [Log2 Scale]')
  plt.ylabel('Time [Log10 ms]')
  plt.title('Compaction Runtime vs. Array Size (non-power-of-two)')
  plt.show()


if __name__ == '__main__':
  main()

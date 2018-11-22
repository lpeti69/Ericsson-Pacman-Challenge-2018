import sys
import random

def read_data():
    # stdin-t olvassuk :)
    first_line_data = list(map(int, sys.stdin.readline().strip().split(" ")))
    second_line_data = sys.stdin.readline().strip().split(" ", 4)
    
    size = (int(second_line_data[0]), int(second_line_data[1]))
    pacman_count = int(second_line_data[2])
    ghost_count = int(second_line_data[3])
    if len(second_line_data) > 4:
        sys.stderr.write("\nGot message: %s\n" % second_line_data[4])
    
    # TODO azert ennel kenyelmesebben is el lehet menteni az adatokat :)
    fields = []
    for i in range(size[0]):
        fields.append(list(sys.stdin.readline())[:size[1]])
    
    pacman_info = []
    for i in range(pacman_count):
        pacman_info.append(sys.stdin.readline().strip().split(" "))
        
    ghost_info = []
    for i in range(ghost_count):
        ghost_info.append(sys.stdin.readline().strip().split(" "))

    return (first_line_data, fields, pacman_info, ghost_info)

possibles = "v<>^"

while True:
    (data, fields, pacmans, ghosts) = read_data()
    
    if data[2] == -1:
        break
    
    # TODO jobb logika, mint a random
    dir = possibles[random.randint(0, len(possibles) - 1)]

    # Ha szeretnetek debug uzenetet kuldeni, akkor megtehetitek.
    # Vigyazzatok, mert maximalisan csak 1024 * 1024 bajtot kaptok vissza
    sys.stderr.write(dir)
    
    # stdout-ra meg mehet ki a megoldas! Mas ne irodjon ki ;)
    sys.stdout.write("%s %s %s %c\n" % (data[0], data[1], data[2], dir))
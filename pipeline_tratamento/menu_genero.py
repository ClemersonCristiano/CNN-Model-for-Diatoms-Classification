def menu_genero():
    
    while True:

        print("\n>>> Digite o numero do gênero da diatomácea e pressione Enter: | [1] - Encyonema | [2] Eunotia | [3] Gomphonema | [4] Navicula | [5] Pinnularia | [6] Digitar outro gênero |")
        genero_input = input(">>> Opção: ")
        
        if genero_input:
            
            if genero_input == "1":
                return "Encyonema"
                
            elif genero_input == "2":
                return "Eunotia"
                
            elif genero_input == "3":
                return "Gomphonema"
                
            elif genero_input == "4":
                return "Navicula"
                
            elif genero_input == "5":
                return "Pinnularia"
                
            elif genero_input == "6":
                print("+-------------------------------------------+")
                genero_result = input("\nDigite o gênero desejado: ").strip()
                
                if not genero_result:
                    print("\n[AVISO] Gênero não informado. Tente novamente.\n")
                    continue
                
                return genero_result
        
            else:
                print("\n[ERRO] Opção inválida. Tente novamente.\n")
                continue
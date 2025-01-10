
import pygame as p
import ChessEngine

WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}


def loadImages():
    pieces = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK"]
    for piece in pieces:
        IMAGES[piece] = p.image.load("images/" + piece + ".png")


def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = ChessEngine.Gamestate()
    validMoves = gs.getValidMoves()
    moveMade = False
    loadImages()
    running = True
    sqSelected = ()
    playerClicks = []
    gameOver = False

    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

            elif e.type == p.MOUSEBUTTONDOWN and not gameOver:
                location = p.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE

                if sqSelected == (row, col):  # Deselect
                    sqSelected = ()
                    playerClicks = []
                else:
                    sqSelected = (row, col)
                    playerClicks.append(sqSelected)

                    # Highlight selected piece's valid moves
                    if len(playerClicks) == 1:
                        piece = gs.board[row][col]
                        if (piece[0] == 'w' and gs.whiteToMove) or (piece[0] == 'b' and not gs.whiteToMove):
                            highlightSquares(screen, gs, sqSelected, validMoves)

                    # Make move
                    if len(playerClicks) == 2:
                        move = ChessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        for validMove in validMoves:
                            if move == validMove:
                                gs.makeMove(move)
                                moveMade = True
                                print(move.getChessNotation())
                                break

                        sqSelected = ()
                        playerClicks = []

                        # If invalid move, keep first click
                        if not moveMade:
                            playerClicks = [sqSelected]

            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:  # Undo when 'z' is pressed
                    gs.undoMove()
                    moveMade = True
                    gameOver = False
                elif e.key == p.K_r:  # Reset when 'r' is pressed
                    gs = ChessEngine.Gamestate()
                    validMoves = gs.getValidMoves()
                    sqSelected = ()
                    playerClicks = []
                    moveMade = False
                    gameOver = False

        if moveMade:
            validMoves = gs.getValidMoves()
            moveMade = False

        drawGameState(screen, gs, sqSelected, validMoves)

        # Draw game over text
        if gameOver:
            drawGameOverText(screen, "Checkmate!" if len(validMoves) == 0 else "Stalemate!")

        clock.tick(MAX_FPS)
        p.display.flip()


def drawGameState(screen, gs, selectedSq, validMoves):
    drawBoard(screen)
    highlightSquares(screen, gs, selectedSq, validMoves)
    drawPieces(screen, gs.board)


def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def highlightSquares(screen, gs, selectedSq, validMoves):
    if selectedSq != ():
        r, c = selectedSq
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):
            # Highlight selected square
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color('blue'))
            screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
            # Highlight valid moves
            s.fill(p.Color('yellow'))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))


def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def drawGameOverText(screen, text):
    font = p.font.SysFont("Helvetica", 32, True, False)
    textObject = font.render(text, False, p.Color("Black"))
    textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH / 2 - textObject.get_width() / 2,
                                                    HEIGHT / 2 - textObject.get_height() / 2)
    screen.blit(textObject, textLocation)


if __name__ == "__main__":
    main()

    #
    # import pygame as p
    # # from pygame.examples.sprite_texture import running
    #
    # import ChessEngine
    #
    # WIDTH = HEIGHT = 512
    # DIMENSION = 8  #dimentions of a board are 8x8
    # SQ_SIZE = HEIGHT // DIMENSION
    # MAX_FPS = 15 # for animantions
    # IMAGES = {}
    #
    #
    # def loadImages():
    #     pieces= ["wP", "wR", "wN", "wB", "wQ", "wK","bP", "bR", "bN", "bB", "bQ", "bK"]
    #     for piece in pieces:
    #         IMAGES[piece] = p.image.load("images/"+piece+".png")
    #
    #
    #
    # '''
    # The main driver of the code. Handle user input and updating the graphics
    #
    # '''
    #
    #
    # def main():
    #     p.init()
    #     screen = p.display.set_mode((WIDTH,HEIGHT))
    #     clock = p.time.Clock()
    #     screen.fill(p.Color("white"))
    #     gs=ChessEngine.Gamestate()
    #     validMoves = gs.getValidMoves()
    #     moveMade = False # flag variable for when a move is made
    #     # print(gs.board)
    #     loadImages()
    #     running = True
    #     sqSelected =() #no sqare is selected, keep track of the last click of the user.(tuple: (row,col))
    #     playerClicks = [] # keep track of pllayer clicks( two tuples:[(6,4),(4,4)]
    #     while running:
    #         for e in p.event.get():
    #             if e.type == p.QUIT:
    #                 running = False
    #             elif e.type == p.MOUSEBUTTONDOWN:
    #                 location = p.mouse.get_pos() # coordinnate of mouse(x,y)
    #                 col = location[0] // SQ_SIZE
    #                 row = location[1] // SQ_SIZE
    #                 if sqSelected == (row,col):  # if the user clicked same square twice
    #                     sqSelected = () # deselect
    #                     playerClicks = [] # clear player clicks
    #
    #                 else:
    #                     sqSelected = (row,col)
    #                     playerClicks.append(sqSelected)  # apped for both first and second clicks
    #                 if len(playerClicks) == 2: #after second click
    #                     move = ChessEngine.Move(playerClicks[0],playerClicks[1], gs.board)
    #                     print(move.getChessNotation())
    #
    #
    #                     gs.makeMove(move)
    #                     moveMade = True
    #
    #                     sqSelected = () # reset user clicks
    #                     playerClicks = []
    #
    #             #key handler
    #             elif e.type == p.KEYDOWN :
    #                 if e.key == p.K_z: #undo when z is pressed
    #                     gs.undoMove()
    #                     moveMade  = True
    #         if moveMade:
    #             validMoves = gs.getValidMoves()
    #             moveMade = False
    #
    #
    #
    #
    #
    #
    #         drawGameState(screen,gs)
    #         clock.tick(MAX_FPS)
    #         p.display.flip()
    #
    #
    # def drawGameState(screen,gs):
    #     drawBoard(screen) # draw sq on board
    #     drawPieces(screen,gs.board) # draw pieces on top of those sq
    #
    # '''
    # Draw the sq on the board. The top left sq is always light
    # '''
    # def drawBoard(screen):
    #     colors = [p.Color("white"),p.Color("gray")]
    #     for r in range(DIMENSION):
    #         for c in range(DIMENSION):
    #             color = colors[((r+c) % 2)]
    #             p.draw.rect(screen,color,p.Rect(c*SQ_SIZE,r*SQ_SIZE,SQ_SIZE,SQ_SIZE))
    #
    #
    # '''
    # '''
    # def drawPieces(screen,board):
    #     for r in range(DIMENSION):
    #         for c in range(DIMENSION):
    #             piece = board[r][c]
    #             if piece != "--":
    #                 screen.blit(IMAGES[piece],p.Rect(c*SQ_SIZE,r*SQ_SIZE,SQ_SIZE,SQ_SIZE))
    #
    #
    #
    # if __name__ == "__main__":
    #     main()
module github.com/tmc/mlx-go-sia

go 1.26.3

require (
	github.com/tmc/localtinker v0.0.0-00010101000000-000000000000
	github.com/tmc/mlx-go v0.0.0-20260529002859-5e7596ef337d
	github.com/tmc/mlx-go-experiments v0.0.0
	github.com/tmc/mlx-go-lm v0.0.0-20260430055726-ce4efcbadf01
)

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/tmc/modelir v0.1.2-0.20260517090425-24c01509645e // indirect
	golang.org/x/image v0.38.0 // indirect
)

replace github.com/tmc/mlx-go-experiments => ../../mlx-go-experiments

replace github.com/tmc/localtinker => ../../localtinker

replace github.com/tmc/mlx-go-lm => ../../mlx-go-lm

replace github.com/tmc/mlx-go => ../../mlx-go

replace github.com/tmc/mlx-go/examples/mlx-go-ccl => ../mlx-go-ccl

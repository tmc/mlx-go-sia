module github.com/tmc/mlx-go-sia

go 1.26.3

require (
	github.com/tmc/localtinker v0.0.0-00010101000000-000000000000
	github.com/tmc/mlx-go v0.0.0-20260430055908-2a38bf0f0098
	github.com/tmc/mlx-go-experiments v0.0.0
)

require github.com/ebitengine/purego v0.10.0 // indirect

replace github.com/tmc/mlx-go-experiments => ../../mlx-go-experiments

replace github.com/tmc/localtinker => ../../localtinker

replace github.com/tmc/mlx-go-lm => ../../mlx-go-lm

replace github.com/tmc/mlx-go => ../../mlx-go

replace github.com/tmc/mlx-go/examples/mlx-go-ccl => ../mlx-go-ccl

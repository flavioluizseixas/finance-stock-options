import numpy as np

def approx_long_put_value_at_short_expiry(ST: np.ndarray, K: float, premium_long: float, t_short: float, t_long: float) -> np.ndarray:
    '''
    Aproxima o valor da put longa na data do vencimento curto.
    Sem recálculo BSM: decompõe premium_long em (intrínseco + extrínseco) e reduz extrínseco por sqrt(tempo_restante/tempo_total).

    - ST: preço no vencimento curto
    - t_short, t_long: em anos, como no banco (t_years). Precisam ser coerentes.
    '''
    intrinsic = np.maximum(K - ST, 0.0)
    extr0 = np.maximum(premium_long - intrinsic, 0.0)

    rem = max(t_long - t_short, 0.0)
    if t_long <= 0:
        factor = 0.0
    else:
        factor = np.sqrt(rem / t_long) if rem > 0 else 0.0

    return intrinsic + extr0 * factor

def payoff_booster_puts_approx(
    ST: np.ndarray,
    K: float,
    premium_long: float,
    premium_short: float,
    t_short: float,
    t_long: float
) -> np.ndarray:
    '''
    Booster horizontal (puts):
      - Long put (longo prazo): paga premium_long e mantém valor no vencimento curto (aprox)
      - Short put (curto prazo): recebe premium_short e expira no vencimento curto

    PnL no vencimento curto:
      PnL = (-premium_long + V_long_approx) + (premium_short - intrinsic_short)
    '''
    v_long = approx_long_put_value_at_short_expiry(ST, K, premium_long, t_short, t_long)
    intrinsic_short = np.maximum(K - ST, 0.0)
    return (-premium_long + v_long) + (premium_short - intrinsic_short)

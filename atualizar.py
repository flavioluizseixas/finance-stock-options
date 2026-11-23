"""Atualiza cotações pela linha de comando (pode ser agendado no Agendador de Tarefas do Windows).

    python atualizar.py                  # ativos acompanhados + ativos com operações abertas
    python atualizar.py PETR4 VALE3      # só estes (+ os de operações abertas)
    python atualizar.py --operacoes      # só os ativos com operações abertas (rápido)
    python atualizar.py --universo       # antes, reavalia o ranking de liquidez dos ativos com opções

O opcoes.net.br limita requisições por IP: a coleta espera ~5 s entre chamadas e, se o site pedir
uma espera longa, para e salva o que já coletou (rode de novo mais tarde).
"""

from __future__ import annotations

import argparse
import sys

from opcoes import dados


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ativos", nargs="*", help="códigos dos ativos-objeto (ex.: PETR4 VALE3)")
    ap.add_argument("--universo", action="store_true", help="reavalia o ranking de liquidez dos ativos com opções")
    ap.add_argument("--operacoes", action="store_true", help="só os ativos com operações abertas")
    args = ap.parse_args()

    def progresso(frac: float, msg: str) -> None:
        print(f"[{frac:5.0%}] {msg}", flush=True)

    if args.universo:
        uni = dados.atualizar_universo(progresso)
        medidos = int(uni["volume_fin"].notna().sum())
        print(f"Universo: {len(uni)} ativos com opções em aberto; volume medido em {medidos}.")
        if dados.meta().get("aviso_universo"):
            print(f"  aviso: {dados.meta()['aviso_universo']}", file=sys.stderr)
    res = dados.atualizar_mercado([a.upper() for a in args.ativos] or None, progresso, somente_operacoes=args.operacoes)
    print(f"Pregão {res.get('sessao')}: {res['linhas']} séries de {len(res['ativos'])} ativos.")
    for chave, erro in (res.get("erros") or {}).items():
        print(f"  aviso {chave}: {erro}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
